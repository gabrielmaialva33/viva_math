//// Fast Fourier Transform over `viva_math/complex`.
////
//// Implements the recursive Cooley-Tukey radix-2 FFT. Inputs to `fft` and
//// `ifft` must have power-of-two length; use `pad_to_power_of_two` when a
//// signal needs zero-padding first.
////
//// ## Examples
////
//// ```gleam
//// import viva_math/complex
//// import viva_math/fft
////
//// fft.fft([
////   complex.real(1.0),
////   complex.zero(),
////   complex.zero(),
////   complex.zero(),
//// ])
//// // -> [1+0i, 1+0i, 1+0i, 1+0i]
//// ```

import gleam/list
import viva_math/complex.{type Complex}
import viva_math/constants

/// Forward FFT.
///
/// The signal length must be a power of two. Invalid lengths panic via
/// `fft_size`, matching the total return type requested for this module.
pub fn fft(signal: List(Complex)) -> List(Complex) {
  let _ = fft_size(signal)
  fft_unchecked(signal)
}

/// Inverse FFT using `IFFT(x) = conj(FFT(conj(x))) / N`.
///
/// The spectrum length must be a power of two. The final result is normalized
/// by `N`, so `ifft(fft(signal))` recovers the original signal up to floating
/// point round-off.
pub fn ifft(spectrum: List(Complex)) -> List(Complex) {
  let n = fft_size(spectrum)
  let scale = 1.0 /. int_to_float(n)

  spectrum
  |> list.map(complex.conjugate)
  |> fft_unchecked
  |> list.map(fn(z) { complex.scale(complex.conjugate(z), scale) })
}

/// Return the signal length if it is a power of two.
///
/// Panics for empty or non-power-of-two inputs because the public API returns
/// `Int` directly.
pub fn fft_size(signal: List(Complex)) -> Int {
  let n = list.length(signal)
  case is_power_of_two(n) {
    True -> n
    False -> panic as "FFT input length must be a non-zero power of two"
  }
}

/// Pad a signal with complex zeros up to the next power of two.
///
/// Already valid lengths are returned unchanged; an empty signal remains empty.
pub fn pad_to_power_of_two(signal: List(Complex)) -> List(Complex) {
  let n = list.length(signal)
  case n {
    0 -> []
    _ -> {
      let target = next_power_of_two(n)
      list.append(signal, list.repeat(complex.zero(), target - n))
    }
  }
}

fn fft_unchecked(signal: List(Complex)) -> List(Complex) {
  case signal {
    [] -> []
    [_] -> signal
    _ -> {
      let #(even, odd) = split_even_odd(signal, [], [])
      combine(
        fft_unchecked(even),
        fft_unchecked(odd),
        0,
        list.length(signal),
        [],
        [],
      )
    }
  }
}

fn split_even_odd(
  signal: List(Complex),
  evens: List(Complex),
  odds: List(Complex),
) -> #(List(Complex), List(Complex)) {
  case signal {
    [] -> #(list.reverse(evens), list.reverse(odds))
    [even] -> #(list.reverse([even, ..evens]), list.reverse(odds))
    [even, odd, ..rest] -> split_even_odd(rest, [even, ..evens], [odd, ..odds])
  }
}

fn combine(
  evens: List(Complex),
  odds: List(Complex),
  k: Int,
  n: Int,
  low: List(Complex),
  high: List(Complex),
) -> List(Complex) {
  case evens, odds {
    [], [] -> list.append(list.reverse(low), list.reverse(high))
    [e, ..even_rest], [o, ..odd_rest] -> {
      let angle = 0.0 -. constants.tau *. int_to_float(k) /. int_to_float(n)
      let twiddle = complex.from_polar(1.0, angle)
      let t = complex.mul(twiddle, o)

      combine(even_rest, odd_rest, k + 1, n, [complex.add(e, t), ..low], [
        complex.sub(e, t),
        ..high
      ])
    }
    _, _ -> list.append(list.reverse(low), list.reverse(high))
  }
}

fn is_power_of_two(n: Int) -> Bool {
  case n {
    n if n <= 0 -> False
    1 -> True
    n if n % 2 != 0 -> False
    _ -> is_power_of_two(n / 2)
  }
}

fn next_power_of_two(n: Int) -> Int {
  next_power_of_two_loop(n, 1)
}

fn next_power_of_two_loop(n: Int, acc: Int) -> Int {
  case acc >= n {
    True -> acc
    False -> next_power_of_two_loop(n, acc * 2)
  }
}

@external(erlang, "erlang", "float")
fn int_to_float(n: Int) -> Float
