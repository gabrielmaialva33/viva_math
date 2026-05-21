import gleeunit/should
import test_support.{is_close, tight, transcendental}
import viva_math/constants
import viva_math/distributions
import viva_math/random
import viva_math/scalar

pub fn gaussian_closed_form_values_test() {
  let g = distributions.Gaussian(mean: 1.0, stddev: 2.0)

  distributions.gaussian_pdf(g, 1.0)
  |> is_close(constants.inv_sqrt_2pi /. 2.0, tight)
  |> should.be_true

  distributions.gaussian_log_pdf(g, 1.0)
  |> is_close(scalar.ln(constants.inv_sqrt_2pi /. 2.0), tight)
  |> should.be_true

  distributions.gaussian_cdf(g, 1.0)
  |> is_close(0.5, tight)
  |> should.be_true

  distributions.standard_normal()
  |> should.equal(distributions.Gaussian(mean: 0.0, stddev: 1.0))
}

pub fn gaussian_sample_is_finite_test() {
  let #(x, _) =
    distributions.gaussian_sample(
      distributions.Gaussian(mean: 10.0, stddev: 0.5),
      random.from_int(123),
    )
  should.be_true(is_finite(x))
}

pub fn laplace_pdf_peak_and_sample_test() {
  let l = distributions.Laplace(location: 0.0, scale: 2.0)
  distributions.laplace_pdf(l, 0.0)
  |> is_close(0.25, tight)
  |> should.be_true

  let #(sample, _) = distributions.laplace_sample(l, random.from_int(7))
  should.be_true(is_finite(sample))
}

pub fn cauchy_pdf_peak_and_sample_test() {
  let c = distributions.Cauchy(location: -1.0, scale: 0.5)
  distributions.cauchy_pdf(c, -1.0)
  |> is_close(1.0 /. { constants.pi *. 0.5 }, tight)
  |> should.be_true

  let #(sample, _) = distributions.cauchy_sample(c, random.from_int(7))
  should.be_true(is_finite(sample))
}

pub fn bernoulli_pmf_and_sample_test() {
  let b = distributions.Bernoulli(p: 0.3)
  distributions.bernoulli_pmf(b, 0) +. distributions.bernoulli_pmf(b, 1)
  |> is_close(1.0, tight)
  |> should.be_true

  distributions.bernoulli_pmf(b, 2)
  |> is_close(0.0, tight)
  |> should.be_true

  let #(always_true, _) =
    distributions.bernoulli_sample(
      distributions.Bernoulli(p: 1.0),
      random.from_int(1),
    )
  let #(always_false, _) =
    distributions.bernoulli_sample(
      distributions.Bernoulli(p: 0.0),
      random.from_int(1),
    )
  always_true |> should.be_true
  always_false |> should.be_false
}

pub fn categorical_pmf_entropy_and_sample_test() {
  let c = distributions.Categorical(probs: [0.2, 0.3, 0.5])
  distributions.categorical_pmf(c, 0)
  +. distributions.categorical_pmf(c, 1)
  +. distributions.categorical_pmf(c, 2)
  |> is_close(1.0, tight)
  |> should.be_true

  distributions.categorical_pmf(c, 10)
  |> is_close(0.0, tight)
  |> should.be_true

  distributions.categorical_entropy(
    distributions.Categorical(probs: [0.5, 0.5]),
  )
  |> is_close(constants.ln_2, transcendental)
  |> should.be_true

  let assert Ok(sample) =
    distributions.categorical_sample(
      distributions.Categorical(probs: [0.0, 1.0, 0.0]),
      random.from_int(99),
    )
  sample.0 |> should.equal(1)
}

fn is_finite(x: Float) -> Bool {
  x <. 1.0e308 && x >. -1.0e308
}
