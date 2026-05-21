const buffer = new ArrayBuffer(8);
const view = new DataView(buffer);

function orderedFloatBits(x) {
  view.setFloat64(0, x, false);
  const bits = view.getBigUint64(0, false);
  const sign = 0x8000000000000000n;

  if ((bits & sign) === 0n) {
    return BigInt.asIntN(64, bits);
  }

  return -1n - BigInt.asIntN(64, bits ^ sign);
}

export function ulp_distance(a, b) {
  const distance = orderedFloatBits(a) - orderedFloatBits(b);
  const absDistance = distance < 0n ? -distance : distance;
  return Number(absDistance);
}
