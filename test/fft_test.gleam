//// Tests for `viva_math/fft`.

import gleam/list
import gleeunit/should
import viva_math/complex.{type Complex, Complex}
import viva_math/fft

const tight: Float = 1.0e-9

fn close_complex(a: Complex, b: Complex) -> Bool {
  complex.is_close(a, b, tight)
}

fn close_list(a: List(Complex), b: List(Complex)) -> Bool {
  list.length(a) == list.length(b)
  && list.all(list.zip(a, b), fn(pair) { close_complex(pair.0, pair.1) })
}

pub fn fft_delta_to_constant_test() {
  let signal = [
    complex.real(1.0),
    complex.zero(),
    complex.zero(),
    complex.zero(),
  ]
  let expected = [
    complex.real(1.0),
    complex.real(1.0),
    complex.real(1.0),
    complex.real(1.0),
  ]

  should.be_true(close_list(fft.fft(signal), expected))
}

pub fn fft_constant_to_delta_test() {
  let signal = [
    complex.real(1.0),
    complex.real(1.0),
    complex.real(1.0),
    complex.real(1.0),
  ]
  let expected = [
    complex.real(4.0),
    complex.zero(),
    complex.zero(),
    complex.zero(),
  ]

  should.be_true(close_list(fft.fft(signal), expected))
}

pub fn ifft_recovers_original_signal_test() {
  let signal = [
    Complex(re: 1.0, im: 0.5),
    Complex(re: -2.0, im: 1.0),
    Complex(re: 0.25, im: -0.75),
    Complex(re: 3.0, im: 0.0),
  ]

  should.be_true(close_list(signal |> fft.fft |> fft.ifft, signal))
}

pub fn pad_to_power_of_two_supports_non_power_input_test() {
  let signal = [complex.real(1.0), complex.real(1.0), complex.real(1.0)]
  let padded = fft.pad_to_power_of_two(signal)
  let expected = [
    complex.real(1.0),
    complex.real(1.0),
    complex.real(1.0),
    complex.zero(),
  ]

  should.equal(fft.fft_size(padded), 4)
  should.be_true(close_list(padded, expected))
  should.be_true(close_list(padded |> fft.fft |> fft.ifft, padded))
}
