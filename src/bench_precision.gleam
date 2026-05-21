//// Benchmarks for the `precision` module.
////
//// Runs with `gleam run -m precision_bench`. Compares naive `list.fold`
//// summation against Neumaier compensated summation across input sizes
//// and condition numbers.

import gleam/io
import gleam/list
import viva_math/precision

pub fn main() {
  io.println("\n=== viva_math precision benchmarks ===\n")

  let sizes = [100, 1000, 10_000, 100_000]

  io.println(
    "[1] Sum accuracy on adversarial input [1, 1e100, 1, -1e100] — exact = 2.0",
  )
  let pathological = [1.0, 1.0e100, 1.0, -1.0e100]
  let naive = list.fold(pathological, 0.0, fn(acc, x) { acc +. x })
  let kahan_val = precision.kahan_sum(pathological)
  let neumaier_val = precision.neumaier_sum(pathological)
  let fsum_val = precision.fsum(pathological)
  io.println("  naive sum     -> " <> float_to_str(naive))
  io.println("  kahan_sum     -> " <> float_to_str(kahan_val))
  io.println("  neumaier_sum  -> " <> float_to_str(neumaier_val))
  io.println("  fsum          -> " <> float_to_str(fsum_val))

  io.println("\n[2] Sum throughput by input size (raw timings)")
  list.each(sizes, fn(n) {
    let xs = build_random(n, 0.123)
    let t0 = monotonic_time_ns()
    let _ = list.fold(xs, 0.0, fn(acc, x) { acc +. x })
    let t1 = monotonic_time_ns()
    let _ = precision.neumaier_sum(xs)
    let t2 = monotonic_time_ns()
    let _ = precision.pairwise_sum(xs)
    let t3 = monotonic_time_ns()
    io.println(
      "  n="
      <> int_to_str(n)
      <> "  naive="
      <> int_to_str(t1 - t0)
      <> "ns  neumaier="
      <> int_to_str(t2 - t1)
      <> "ns  pairwise="
      <> int_to_str(t3 - t2)
      <> "ns",
    )
  })

  io.println("\n[3] Pébay moments — variance + skew + kurtosis in one pass")
  let xs = build_random(10_000, 0.456)
  let t0 = monotonic_time_ns()
  let m = precision.moments_from_list(xs)
  let t1 = monotonic_time_ns()
  let _ = precision.moments_variance(m)
  let _ = precision.moments_skewness(m)
  let _ = precision.moments_excess_kurtosis(m)
  io.println(
    "  10k samples, single-pass Welford+Pébay → " <> int_to_str(t1 - t0) <> "ns",
  )

  io.println("\nDone.\n")
}

fn build_random(n: Int, seed: Float) -> List(Float) {
  build_loop(n, seed, [])
}

fn build_loop(n: Int, seed: Float, acc: List(Float)) -> List(Float) {
  case n <= 0 {
    True -> acc
    False -> {
      // Deterministic pseudo-random in [-1, 1]
      let next = lcg(seed)
      build_loop(n - 1, next, [next, ..acc])
    }
  }
}

fn lcg(x: Float) -> Float {
  let v = float_fmod(x *. 16_807.0 +. 0.123_456_789, 2.0) -. 1.0
  v
}

@external(erlang, "math", "fmod")
@external(javascript, "./viva_math_random_ffi.mjs", "fmod")
fn float_fmod(a: Float, b: Float) -> Float

@external(erlang, "erlang", "monotonic_time")
@external(javascript, "./viva_math_random_ffi.mjs", "monotonic_time_ns")
fn monotonic_time_ns() -> Int

@external(erlang, "erlang", "integer_to_binary")
@external(javascript, "./viva_math_random_ffi.mjs", "int_to_string")
fn int_to_str(n: Int) -> String

@external(erlang, "erlang", "float_to_binary")
@external(javascript, "./viva_math_random_ffi.mjs", "float_to_string")
fn float_to_str(f: Float) -> String
