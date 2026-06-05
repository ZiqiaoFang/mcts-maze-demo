// AZ network: small two-headed MLP. Lives in the browser only.
// Assumes the `tf` global is available (loaded via CDN in index.html).
import { CHANNELS, encodeState } from "./az-encode.js";

// Pick the fastest available TF.js backend for this workload.
// Counterintuitive note: for MCTS we run ~50 simulations per move, each doing
// a single-sample forward pass through an 80k-param MLP. At that batch size,
// WebGL beats WebGPU because WebGPU's per-call overhead doesn't amortize.
// Measured locally on the 10×10 maze: 1.73 ms/predict on WebGL vs 2.97 on
// WebGPU. So we prefer webgl → webgpu → cpu.
export async function setupBackend() {
  if (typeof tf === "undefined") return null;
  const tryBackend = async (name) => {
    try {
      const ok = await tf.setBackend(name);
      if (!ok) return false;
      await tf.ready();
      return tf.getBackend() === name;
    } catch (e) {
      return false;
    }
  };
  if (await tryBackend("webgl")) return "webgl";
  if (await tryBackend("webgpu")) return "webgpu";
  await tryBackend("cpu");
  return tf.getBackend();
}

// Pre-compile model shaders by running one throwaway forward pass.
// Without this, the first real predict() takes 100-500ms while WebGL/WebGPU
// compiles shaders — visible as a freeze on the first click.
export function warmupModel(model, size) {
  const inputDim = size * size * CHANNELS;
  tf.tidy(() => {
    const dummy = tf.zeros([1, inputDim]);
    const [p, v] = model.predict(dummy);
    p.dataSync();
    v.dataSync();
  });
}

// Small CNN with two output heads.
// Input: flat [size*size*CHANNELS] — reshaped internally to [size, size, CHANNELS]
//        so the rest of the pipeline (predict / predictBatch / replay buffer)
//        keeps its flat encoding and we only change the network's first layer.
// Trunk: Conv2D(16, 3×3, same) → Conv2D(16, 3×3, same) → Flatten → Dense(64)
// Heads: p (4, softmax) and v (1, linear)
//
// Why CNN over MLP: an MLP treats every (r,c) cell as an independent feature,
// so "wall to my north" looks like a different signal at (3,4) than at (5,6).
// That makes generalization across maze layouts (domain randomization, novel
// mazes) essentially impossible — the network has to memorize per-position
// answers. Conv2D with padding='same' gives translation-equivariant features;
// "wall to my north" is detected by the same filter wherever it appears.
//
// Why tanh on v: bounds v ∈ [-1, 1], which matches all the z presets
// (Pure win/loss ∈ {-1, +1}, Distance-shaped ∈ [0, 1], Step penalty ∈ [-1, 1]).
// A linear head looks attractive because it has no range mismatch, BUT with
// 4 gradient steps/game and high-variance targets from domain randomization,
// nothing keeps |v| in the z range — we saw value loss diverge to ~5e23
// once the trunk's CNN activations coupled into an unbounded head.
//
// The earlier dead-tanh failure (v parked at ±1, heatmap monochrome) was
// caused by *unbounded weights*, not by tanh itself. With the L2 weight decay
// in trainStep, weights stay small enough that pre-activations stay in
// tanh's responsive zone and saturation doesn't fire. Belt-and-suspenders.
export function createModel(size) {
  const inputDim = size * size * CHANNELS;
  const input = tf.input({ shape: [inputDim] });
  // encodeState lays out (r*size+c)*CHANNELS+ch, which IS channels-last
  // row-major. Reshaping to [size, size, CHANNELS] is a no-op on the data.
  const grid = tf.layers
    .reshape({ targetShape: [size, size, CHANNELS] })
    .apply(input);
  const c1 = tf.layers
    .conv2d({ filters: 16, kernelSize: 3, padding: "same", activation: "relu" })
    .apply(grid);
  const c2 = tf.layers
    .conv2d({ filters: 16, kernelSize: 3, padding: "same", activation: "relu" })
    .apply(c1);
  const flat = tf.layers.flatten().apply(c2);
  const h = tf.layers.dense({ units: 64, activation: "relu" }).apply(flat);
  const pHead = tf.layers
    .dense({ units: 4, activation: "softmax", name: "policy" })
    .apply(h);
  const vHead = tf.layers
    .dense({ units: 1, activation: "tanh", name: "value" })
    .apply(h);
  return tf.model({ inputs: input, outputs: [pHead, vHead] });
}

// Single-state prediction. Returns { p: Float32Array(4), v: number }.
// Allocates and immediately disposes intermediate tensors.
export function predict(model, maze, agentPos) {
  const enc = encodeState(maze, agentPos);
  return tf.tidy(() => {
    const x = tf.tensor2d(enc, [1, enc.length]);
    const [pTensor, vTensor] = model.predict(x);
    const p = pTensor.dataSync().slice(); // copy out of WebGL (already a Float32Array)
    const v = vTensor.dataSync()[0];
    return { p, v };
  });
}

// Batched prediction over many (maze, pos) states sharing the same maze.
// Returns { ps: Float32Array(N*4), vs: Float32Array(N) } in input order.
export function predictBatch(model, maze, positions) {
  const size = maze.size;
  const inputDim = size * size * CHANNELS;
  const N = positions.length;
  const flat = new Float32Array(N * inputDim);
  for (let i = 0; i < N; i++) {
    const row = encodeState(maze, positions[i]);
    flat.set(row, i * inputDim);
  }
  return tf.tidy(() => {
    const x = tf.tensor2d(flat, [N, inputDim]);
    const [pTensor, vTensor] = model.predict(x);
    const ps = Float32Array.from(pTensor.dataSync());
    const vs = Float32Array.from(vTensor.dataSync());
    return { ps, vs };
  });
}

// One gradient step on a single minibatch.
// batch: { states: Float32Array(B*inputDim), pis: Float32Array(B*4), zs: Float32Array(B) }
// Returns { total, policy, value } as plain numbers.
export function trainStep(model, optimizer, batch, inputDim) {
  const B = batch.zs.length;
  const x = tf.tensor2d(batch.states, [B, inputDim]);
  const piTarget = tf.tensor2d(batch.pis, [B, 4]);
  const zTarget = tf.tensor2d(batch.zs, [B, 1]);
  let losses;
  // Weight-decay coefficient. The spec §4.2 (and the original AlphaGo Zero
  // paper) use λ=1e-4, but that's tuned for a stationary self-play
  // distribution with massive batches. Our setup is messier — small replay
  // buffer, Adam (which normalizes per-parameter and damps weak L2
  // gradients), 4 SGD steps per game, and high-variance z under Step penalty
  // / domain randomization. λ=1e-3 puts the inward pull on weights into a
  // range Adam can't drown out, which is what actually keeps the trunk from
  // drifting and the loss from running away.
  const L2_LAMBDA = 1e-3;
  optimizer.minimize(() => {
    const [pPred, vPred] = model.apply(x);
    // value loss: MSE
    const vLoss = tf.losses.meanSquaredError(zTarget, vPred);
    // Policy loss: -mean( Σ_a pi(a) · log(p(a)) ).
    //
    // Clamping pPred to [1e-3, 1] before log is what keeps training stable.
    // A naked `log(pPred + 1e-7)` lets per-sample CE reach -log(1e-7)≈16.1
    // when softmax saturates onto the wrong action — gradients explode and
    // the policy gets stuck in a bad attractor (we observed loss plateauing
    // around 14, dominated by these log-of-tiny spikes). Clipping to 1e-3
    // caps per-sample CE at -log(1e-3)≈6.9, which is still a strong gradient
    // but won't pinball the trunk weights.
    const pClipped = tf.clipByValue(pPred, 1e-3, 1);
    const pLoss = tf
      .neg(tf.sum(tf.mul(piTarget, tf.log(pClipped))))
      .div(tf.scalar(B));
    // L2 weight decay summed across every trainable variable.
    const l2Sum = tf.addN(
      model.trainableWeights.map((w) => tf.sum(tf.square(w.read()))),
    );
    const l2Loss = tf.mul(tf.scalar(L2_LAMBDA), l2Sum);
    const total = tf.add(tf.add(vLoss, pLoss), l2Loss);
    losses = {
      total: total.dataSync()[0],
      policy: pLoss.dataSync()[0],
      value: vLoss.dataSync()[0],
    };
    return total;
  }, /* returnCost */ false);
  x.dispose();
  piTarget.dispose();
  zTarget.dispose();
  return losses;
}
