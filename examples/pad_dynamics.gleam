//// Example: simulating PAD emotional dynamics under stochastic stimuli.
////
//// Copy this file into `src/` and run `gleam run -m pad_dynamics`.

import gleam/io
import gleam/list
import viva_math/attractor
import viva_math/ode
import viva_math/random
import viva_math/vector

pub fn main() {
  io.println("\n=== PAD dynamics under stochastic stimulus ===\n")

  // Initial state: neutral
  let x0 = 0.0

  // Drift: gentle attraction toward x = 0.5 (slight positive valence)
  let drift = fn(_t, x) { 0.2 *. { 0.5 -. x } }

  // Diffusion: constant 0.3 (noise amplitude)
  let diffusion = fn(_t, _x) { 0.3 }

  let seed = random.from_int(7)
  let dt = 0.1
  let steps = 50
  let #(trajectory, _) = ode.integrate_sde(drift, diffusion, 0.0, x0, dt, steps, seed)

  io.println("Time series of valence x(t):")
  list.each(trajectory, fn(pair) {
    let #(t, x) = pair
    io.println("  t=" <> float_str(t) <> "  x=" <> float_str(x))
  })

  // Final state classification along the PAD axis (using only the x component)
  let last = case list.last(trajectory) {
    Ok(p) -> p.1
    Error(_) -> 0.0
  }
  let state = vector.Vec3(last, 0.0, 0.0)
  io.println("\nFinal emotion (nearest Mehrabian attractor): " <> attractor.classify_emotion(state))
}

@external(erlang, "erlang", "float_to_binary")
fn float_str(f: Float) -> String
