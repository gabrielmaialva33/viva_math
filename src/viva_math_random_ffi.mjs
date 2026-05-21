function normalizeSeed(seed) {
  if (seed && typeof seed.state === "number") {
    return seed.state >>> 0;
  }

  return Number(seed) >>> 0;
}

function makeSeed(state) {
  return { algo: "mulberry32", state: state >>> 0 };
}

export function int_to_float(n) {
  return Number(n);
}

export const sqrt = Math.sqrt;
export const exp = Math.exp;
export const log = Math.log;
export const sin = Math.sin;
export const cos = Math.cos;
export const tanh = Math.tanh;
export const pow = Math.pow;
export const tan = Math.tan;
export const asin = Math.asin;
export const acos = Math.acos;
export const atan = Math.atan;
export const atan2 = Math.atan2;
export const log2 = Math.log2;
export const log10 = Math.log10;
export const sinh = Math.sinh;
export const cosh = Math.cosh;
export const trunc = Math.trunc;

function polevl(x, coef) {
  let ans = 0;
  for (const c of coef) {
    ans = ans * x + c;
  }
  return ans;
}

function p1evl(x, coef) {
  let ans = x + coef[0];
  for (let i = 1; i < coef.length; i += 1) {
    ans = ans * x + coef[i];
  }
  return ans;
}

const erfT = [
  9.60497373987051638749e0,
  9.00260197203842689217e1,
  2.23200534594684319226e3,
  7.00332514112805075473e3,
  5.55923013010394962768e4,
];

const erfU = [
  3.35617141647503099647e1,
  5.21357949780152679795e2,
  4.59432382970980127987e3,
  2.26290000613890934246e4,
  4.92673942608635921086e4,
];

const erfcP = [
  2.46196981473530512524e-10,
  5.64189564831068821977e-1,
  7.46321056442269912687e0,
  4.86371970985681366614e1,
  1.96520832956077098242e2,
  5.26445194995477358631e2,
  9.34528527171957607540e2,
  1.02755188689515710272e3,
  5.57535335369399327526e2,
];

const erfcQ = [
  1.32281951154744992508e1,
  8.67072140885989742329e1,
  3.54937778887819891062e2,
  9.75708501743205489753e2,
  1.82390916687909736289e3,
  2.24633760818710981792e3,
  1.65666309194161350182e3,
  5.57535340817727675546e2,
];

const erfcR = [
  5.64189583547755073984e-1,
  1.27536670759978104416e0,
  5.01905042251180477414e0,
  6.16021097993053585195e0,
  7.40974269950448939160e0,
  2.97886665372100240670e0,
];

const erfcS = [
  2.26052863220117276590e0,
  9.39603524938001434673e0,
  1.20489539808096656605e1,
  1.70814450747565897222e1,
  9.60896809063285878198e0,
  3.36907645100081516050e0,
];

export function erf(x) {
  if (Math.abs(x) > 1) {
    return x < 0 ? erfc(-x) - 1 : 1 - erfc(x);
  }

  const z = x * x;
  return x * polevl(z, erfT) / p1evl(z, erfU);
}

export function erfc(x) {
  if (x < 0) {
    return 2 - erfc(-x);
  }
  if (x < 1) {
    return 1 - erf(x);
  }

  const z = Math.exp(-x * x);
  if (x < 8) {
    return z * polevl(x, erfcP) / p1evl(x, erfcQ);
  }
  return z * polevl(x, erfcR) / p1evl(x, erfcS);
}

export function fmod(x, y) {
  return x - Math.trunc(x / y) * y;
}

export function bit_size(bits) {
  if (bits && typeof bits.bitSize === "number") {
    return bits.bitSize;
  }
  if (bits && typeof bits.bitLength === "number") {
    return bits.bitLength;
  }
  if (bits && typeof bits.byteLength === "number") {
    return bits.byteLength * 8;
  }
  if (typeof bits === "string") {
    return bits.length * 8;
  }
  if (Array.isArray(bits)) {
    return bits.length * 8;
  }
  return 0;
}

export function monotonic_time_ns() {
  return Math.trunc(performance.now() * 1_000_000);
}

export function int_to_string(n) {
  return String(n);
}

export function float_to_string(f) {
  return String(f);
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
