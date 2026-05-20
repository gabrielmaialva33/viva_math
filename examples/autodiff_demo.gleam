//// Example: exact gradient via forward-mode autodiff.
////
//// Copy into `src/` and `gleam run -m autodiff_demo`.

import gleam/io
import viva_math/autodiff as ad

pub fn main() {
  io.println("\n=== Autodiff: exact gradient at a point ===\n")

  // f(x) = sin(x²) · GELU(x)
  let f = fn(x: ad.Dual) -> ad.Dual {
    let x_sq = ad.mul(x, x)
    let sin_term = ad.sin(x_sq)
    let gelu_term = ad.gelu(x)
    ad.mul(sin_term, gelu_term)
  }

  let xs = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
  io.println("    x         f(x)           f'(x)")
  io.println("  -------   -----------     -----------")
  loop(xs, f)
}

fn loop(xs: List(Float), f: fn(ad.Dual) -> ad.Dual) {
  case xs {
    [] -> Nil
    [x, ..rest] -> {
      let #(value, grad) = ad.value_and_grad(f, x)
      io.println(
        "  "
        <> float_str(x)
        <> "    "
        <> float_str(value)
        <> "    "
        <> float_str(grad),
      )
      loop(rest, f)
    }
  }
}

@external(erlang, "erlang", "float_to_binary")
fn float_str(f: Float) -> String
