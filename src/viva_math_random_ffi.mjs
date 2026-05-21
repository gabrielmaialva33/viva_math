function normalizeSeed(seed) {
  if (seed && typeof seed.state === "number") {
    return seed.state >>> 0;
  }

  return Number(seed) >>> 0;
}

function makeSeed(state) {
  return { algo: "mulberry32", state: state >>> 0 };
}

function nextMulberry32(seed) {
  let state = normalizeSeed(seed);
  state = (state + 0x6d2b79f5) >>> 0;

  let t = state;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

  const value = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  return [value, makeSeed(state)];
}

export function seed_default(seed) {
  return makeSeed(seed);
}

export function seed_with_algo(_algo, seed) {
  return makeSeed(seed);
}

export function uniform_real(state) {
  return nextMulberry32(state);
}

export function uniform_int(n, state) {
  const [u, nextState] = nextMulberry32(state);
  return [Math.floor(u * n) + 1, nextState];
}

export function normal_standard(state) {
  const [u1Raw, state1] = nextMulberry32(state);
  const [u2, state2] = nextMulberry32(state1);
  const u1 = Math.max(u1Raw, Number.MIN_VALUE);
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return [z0, state2];
}

export function normal_with(mu, sigma, state) {
  const [z, nextState] = normal_standard(state);
  return [mu + sigma * z, nextState];
}

export function jump(state) {
  // Mulberry32 has no cheap jump-ahead operation; JS keeps this as a no-op.
  return state;
}
