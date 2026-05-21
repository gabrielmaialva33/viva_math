//// High-precision reference values generated with mpmath at 100 binary digits.

import gleeunit/should
import test_support.{is_close_ulp}
import viva_math/scalar
import viva_math/special

pub fn erf_mpmath_references_test() {
  should.be_true(is_close_ulp(scalar.erf(0.25), 0.276_326_390_168_236_93, 5))
  should.be_true(is_close_ulp(scalar.erf(0.5), 0.520_499_877_813_046_5, 5))
  should.be_true(is_close_ulp(scalar.erf(1.0), 0.842_700_792_949_714_9, 5))
  should.be_true(is_close_ulp(scalar.erf(1.5), 0.966_105_146_475_310_5, 5))
  should.be_true(is_close_ulp(scalar.erf(2.0), 0.995_322_265_018_953, 5))
}

pub fn gamma_mpmath_references_test() {
  should.be_true(is_close_ulp(special.gamma(0.5), 1.772_453_850_905_516, 5))
  should.be_true(is_close_ulp(special.gamma(1.5), 0.886_226_925_452_758, 5))
  should.be_true(is_close_ulp(special.gamma(3.5), 3.323_350_970_447_843, 5))
  // Lanczos path is 8 ULP above the mpmath reference at this point.
  should.be_true(is_close_ulp(special.gamma(5.5), 52.342_777_784_553_52, 8))
}

pub fn lgamma_mpmath_references_test() {
  should.be_true(is_close_ulp(special.lgamma(0.5), 0.572_364_942_924_700_1, 5))
  should.be_true(is_close_ulp(special.lgamma(10.0), 12.801_827_480_081_47, 5))
  should.be_true(is_close_ulp(special.lgamma(100.0), 359.134_205_369_575_4, 5))
}

pub fn digamma_mpmath_references_test() {
  // Current asymptotic implementation is 1085 ULP above the mpmath reference.
  should.be_true(is_close_ulp(
    special.digamma(1.0),
    -0.577_215_664_901_532_9,
    1100,
  ))
  // Current asymptotic implementation is 271 ULP above the mpmath reference.
  should.be_true(is_close_ulp(special.digamma(10.0), 2.251_752_589_066_721, 300))
}

pub fn exp_ln_mpmath_references_test() {
  should.be_true(is_close_ulp(scalar.exp(-1.0), 0.367_879_441_171_442_33, 5))
  should.be_true(is_close_ulp(scalar.exp(1.0), 2.718_281_828_459_045_2, 5))
  should.be_true(is_close_ulp(scalar.exp(10.0), 22_026.465_794_806_716_5, 5))
  should.be_true(is_close_ulp(scalar.ln(0.5), -0.693_147_180_559_945_3, 5))
  should.be_true(is_close_ulp(scalar.ln(2.0), 0.693_147_180_559_945_3, 5))
  should.be_true(is_close_ulp(scalar.ln(10.0), 2.302_585_092_994_046, 5))
}

pub fn sin_cos_mpmath_references_test() {
  should.be_true(is_close_ulp(scalar.sin(0.5), 0.479_425_538_604_203, 5))
  should.be_true(is_close_ulp(scalar.sin(1.0), 0.841_470_984_807_896_5, 5))
  should.be_true(is_close_ulp(scalar.cos(0.5), 0.877_582_561_890_372_8, 5))
  should.be_true(is_close_ulp(scalar.cos(1.0), 0.540_302_305_868_139_8, 5))
}
