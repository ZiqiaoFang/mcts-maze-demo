// AZ network: small two-headed MLP. Lives in the browser only.
// Assumes the `tf` global is available (loaded via CDN in index.html).
import { CHANNELS, encodeState } from "./az-encode.js";

// Build a 2-hidden-layer MLP with two output heads.
// Input: flat [size*size*CHANNELS]
// Hidden: 64 → 64 (relu)
// Heads: p (4, softmax) and v (1, tanh)
export function createModel(size) {
  const inputDim = size * size * CHANNELS;
  const input = tf.input({ shape: [inputDim] });
  const h1 = tf.layers.dense({ units: 64, activation: "relu" }).apply(input);
  const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1);
  const pHead = tf.layers
    .dense({ units: 4, activation: "softmax", name: "policy" })
    .apply(h2);
  const vHead = tf.layers
    .dense({ units: 1, activation: "tanh", name: "value" })
    .apply(h2);
  return tf.model({ inputs: input, outputs: [pHead, vHead] });
}

// Single-state prediction. Returns { p: Float32Array(4), v: number }.
// Allocates and immediately disposes intermediate tensors.
export function predict(model, maze, agentPos) {
  const enc = encodeState(maze, agentPos);
  return tf.tidy(() => {
    const x = tf.tensor2d(enc, [1, enc.length]);
    const [pTensor, vTensor] = model.predict(x);
    const p = pTensor.dataSync().slice(); // copy out of WebGL
    const v = vTensor.dataSync()[0];
    return { p: Float32Array.from(p), v };
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
  optimizer.minimize(() => {
    const [pPred, vPred] = model.apply(x);
    // value loss: MSE
    const vLoss = tf.losses.meanSquaredError(zTarget, vPred);
    // policy loss: -sum(pi * log(p)) averaged over batch
    const eps = tf.scalar(1e-8);
    const pLoss = tf
      .neg(tf.sum(tf.mul(piTarget, tf.log(tf.add(pPred, eps)))))
      .div(tf.scalar(B));
    const total = tf.add(vLoss, pLoss);
    losses = {
      total: total.dataSync()[0],
      policy: pLoss.dataSync()[0],
      value: vLoss.dataSync()[0],
    };
    eps.dispose();
    return total;
  }, /* returnCost */ false);
  x.dispose();
  piTarget.dispose();
  zTarget.dispose();
  return losses;
}
