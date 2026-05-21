import gleam/list
import gleeunit/should
import test_support.{is_close, is_close_vec3, tight}
import viva_math/free_energy
import viva_math/vector.{Vec3}

pub fn free_energy_components_and_state_test() {
  let expected = Vec3(1.0, 0.0, 0.0)
  let actual = Vec3(2.0, 0.0, 0.0)
  let baseline = Vec3(0.0, 0.0, 0.0)
  let thresholds = free_energy.FeelingThresholds(mean: 2.0, std_dev: 0.5)

  free_energy.precision_weighted_prediction_error(expected, actual, 2.0)
  |> is_close(2.0, tight)
  |> should.be_true

  free_energy.complexity(actual, baseline, 2.0)
  |> is_close(1.0, tight)
  |> should.be_true

  free_energy.free_energy(expected, actual, baseline, 2.0, 2.0)
  |> is_close(3.0, tight)
  |> should.be_true

  let state =
    free_energy.compute_state(expected, actual, baseline, 2.0, 2.0, thresholds)
  state.free_energy |> is_close(3.0, tight) |> should.be_true
  state.prediction_error |> is_close(2.0, tight) |> should.be_true
  state.complexity |> is_close(1.0, tight) |> should.be_true
  state.precision |> is_close(2.0, tight) |> should.be_true
  state.feeling |> should.equal(free_energy.Overwhelmed)
}

pub fn surprise_and_variational_bound_test() {
  free_energy.surprise(2.0, 2.0, 0.5)
  |> is_close(0.0, tight)
  |> should.be_true

  free_energy.surprise(1.0, 2.0, 2.0)
  |> is_close(0.125, tight)
  |> should.be_true

  free_energy.variational_bound(1.0, 0.25)
  |> is_close(0.25, tight)
  |> should.be_true

  free_energy.variational_bound(0.0, 1.0)
  |> is_close(101.0, tight)
  |> should.be_true
}

pub fn belief_and_bpc_updates_sum_precision_test() {
  free_energy.belief_update(2.0, 10.0, 3.0, 1.0)
  |> is_close(4.0, tight)
  |> should.be_true

  free_energy.belief_update(2.0, 10.0, 0.0, 0.0)
  |> is_close(2.0, tight)
  |> should.be_true

  let prior =
    free_energy.GaussianBelief(mean: Vec3(0.0, 0.0, 0.0), precision: 2.0)
  let posterior = free_energy.bpc_update(prior, Vec3(2.0, 4.0, 6.0), 2.0)
  posterior.precision |> is_close(4.0, tight) |> should.be_true
  posterior.mean |> is_close_vec3(Vec3(1.0, 2.0, 3.0), tight) |> should.be_true

  free_energy.bpc_precision_update(3.0, 10.0, 0)
  |> is_close(3.0, tight)
  |> should.be_true
}

pub fn precision_estimates_and_weighted_errors_test() {
  free_energy.estimate_precision([])
  |> is_close(1.0, tight)
  |> should.be_true

  free_energy.estimate_precision([1.0, 1.0, 1.0])
  |> is_close(100.0, tight)
  |> should.be_true

  free_energy.estimate_precision([0.0, 2.0])
  |> is_close(1.0, tight)
  |> should.be_true

  free_energy.precision_weighted_error_vec(
    Vec3(1.0, 2.0, 3.0),
    Vec3(2.0, 0.0, 4.0),
    Vec3(10.0, 1.0, 0.5),
  )
  |> is_close(14.5, tight)
  |> should.be_true

  free_energy.active_inference_delta(
    Vec3(1.0, 1.0, 1.0),
    Vec3(3.0, 5.0, 7.0),
    0.25,
  )
  |> is_close_vec3(Vec3(0.5, 1.0, 1.5), tight)
  |> should.be_true
}

pub fn feeling_classification_and_threshold_updates_test() {
  free_energy.classify_feeling(0.05) |> should.equal(free_energy.Homeostatic)
  free_energy.classify_feeling(0.2) |> should.equal(free_energy.Surprised)
  free_energy.classify_feeling(1.0) |> should.equal(free_energy.Alarmed)
  free_energy.classify_feeling(2.0) |> should.equal(free_energy.Overwhelmed)

  let thresholds = free_energy.FeelingThresholds(mean: 1.0, std_dev: 0.25)
  free_energy.classify_feeling_normalized(0.7, thresholds)
  |> should.equal(free_energy.Homeostatic)
  free_energy.classify_feeling_normalized(0.9, thresholds)
  |> should.equal(free_energy.Surprised)
  free_energy.classify_feeling_normalized(1.1, thresholds)
  |> should.equal(free_energy.Alarmed)
  free_energy.classify_feeling_normalized(1.4, thresholds)
  |> should.equal(free_energy.Overwhelmed)

  let updated = free_energy.update_thresholds(thresholds, 2.0, 0.5)
  updated.mean |> is_close(1.5, tight) |> should.be_true
  should.be_true(updated.std_dev >=. 0.01)
}

pub fn expected_free_energy_policy_selection_and_posterior_test() {
  let preferred = Vec3(0.0, 0.0, 0.0)
  let g = free_energy.expected_free_energy(Vec3(1.0, 0.0, 0.0), preferred, 0.25)
  g.epistemic |> is_close(0.25, tight) |> should.be_true
  g.pragmatic |> is_close(1.0, tight) |> should.be_true
  g.total |> is_close(1.25, tight) |> should.be_true

  free_energy.generalized_free_energy(Vec3(1.0, 2.0, 0.0), preferred, 0.5)
  |> is_close(5.5, tight)
  |> should.be_true

  let policies = [
    #("far", Vec3(2.0, 0.0, 0.0), 0.0),
    #("near", Vec3(0.1, 0.0, 0.0), 0.0),
    #("uncertain", Vec3(0.0, 0.0, 0.0), 1.0),
  ]
  let assert Ok(best) = free_energy.select_policy(policies, preferred)
  best.0 |> should.equal("near")
  best.1.total |> is_close(0.01, tight) |> should.be_true

  let posterior = free_energy.policy_posterior(policies, preferred, 2.0)
  posterior_sum(posterior)
  |> is_close(1.0, tight)
  |> should.be_true
}

pub fn hierarchical_prediction_errors_zero_when_equal_test() {
  let layer =
    free_energy.HierarchicalLayer(
      mu: Vec3(1.0, 2.0, 3.0),
      precision: 2.0,
      prior_precision: 1.0,
    )
  let h = free_energy.Hierarchical(layers: [layer, layer, layer])

  let errors = free_energy.hierarchical_errors(h)
  should.equal(list.length(errors), 2)
  list.all(errors, fn(e) { is_close_vec3(e, Vec3(0.0, 0.0, 0.0), tight) })
  |> should.be_true

  let decoded_errors = free_energy.hierarchical_errors_with(h, fn(v) { v })
  should.equal(list.length(decoded_errors), 2)
  list.all(decoded_errors, fn(e) {
    is_close_vec3(e, Vec3(0.0, 0.0, 0.0), tight)
  })
  |> should.be_true

  free_energy.hierarchical_free_energy(h)
  |> is_close(0.0, tight)
  |> should.be_true

  let meta = free_energy.meta_prediction_errors(h)
  should.equal(list.length(meta), 1)
  list.all(meta, fn(e) { is_close_vec3(e, Vec3(0.0, 0.0, 0.0), tight) })
  |> should.be_true
}

pub fn hierarchical_inference_stable_for_zero_error_test() {
  let layer =
    free_energy.HierarchicalLayer(
      mu: Vec3(0.5, -0.25, 0.75),
      precision: 1.0,
      prior_precision: 1.0,
    )
  let h = free_energy.Hierarchical(layers: [layer, layer])

  let stepped = free_energy.hierarchical_inference_step(h, 0.1)
  let inferred = free_energy.hierarchical_infer(h, 0.1, 5)

  should.equal(stepped.layers, h.layers)
  should.equal(inferred.layers, h.layers)
}

fn posterior_sum(posterior: List(#(a, Float))) -> Float {
  list.fold(posterior, 0.0, fn(acc, pair) { acc +. pair.1 })
}
