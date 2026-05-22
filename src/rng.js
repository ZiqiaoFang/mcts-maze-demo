// Mulberry32: small, seedable PRNG. Returns a function that yields [0, 1).
export function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// Convenience: pick a uniform element from an array using a given rng().
export function pick(arr, rng) {
  return arr[Math.floor(rng() * arr.length)];
}
