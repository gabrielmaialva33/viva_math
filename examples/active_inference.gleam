//// Example: minimal active inference policy selection via Expected Free Energy.
////
//// Copy into `src/` and `gleam run -m active_inference`.

import gleam/io
import gleam/list
import viva_math/free_energy
import viva_math/vector

pub fn main() {
  io.println("\n=== Active Inference policy selection ===\n")

  // Preferred outcome: positive, calm, dominant.
  let goal = vector.Vec3(0.7, -0.3, 0.5)

  // Three candidate policies, each labelled with predicted outcome + uncertainty.
  let policies = [
    #("approach",   vector.Vec3(0.6, -0.2, 0.4), 0.1),
    #("withdraw",   vector.Vec3(0.1, 0.1, -0.3),  0.5),
    #("freeze",     vector.Vec3(0.0, 0.0, 0.0),  0.8),
  ]

  io.println("Policy posterior (softmax of -β · G):")
  let posterior = free_energy.policy_posterior(policies, goal, 2.0)
  list.each(posterior, fn(p) {
    io.println("  " <> p.0 <> "  p=" <> float_str(p.1))
  })

  case free_energy.select_policy(policies, goal) {
    Ok(pair) -> io.println("\nBest policy: " <> pair.0)
    Error(_) -> io.println("\nNo policies provided")
  }
}

@external(erlang, "erlang", "float_to_binary")
fn float_str(f: Float) -> String
