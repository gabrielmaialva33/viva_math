//// Reverse-mode automatic differentiation via a computation tape.
////
//// Forward-mode AD (`viva_math/autodiff`) is optimal when the input
//// dimension is small relative to the output (e.g. one-variable
//// gradients). Reverse-mode AD ("backprop") is optimal for the opposite
//// case: scalar output, high-dimensional input — exactly the regime of
//// neural networks and gradient-based MAP inference.
////
//// ## How it works
////
//// 1. The forward pass builds a directed acyclic graph (the "tape") that
////    records every elementary operation and its operands.
//// 2. The backward pass walks the tape in reverse, applying the chain
////    rule to accumulate `∂output/∂node` for every node.
////
//// All inputs that share a tape are differentiated in a single backward
//// pass, regardless of how many there are — that's why reverse-mode AD is
//// O(1) in input dimension for gradient computation.
////
//// ## Limitations of this implementation
////
//// - Single-output, scalar gradients only. For Jacobians of vector-valued
////   functions, use `viva_math/autodiff.jacobian` (forward) or call this
////   multiple times.
//// - The tape is a `Dict(Int, ...)` for clarity; production tools use
////   linear arrays for cache locality. Adequate for graphs ≤ 10⁴ nodes.
////
//// ## Example
////
//// ```gleam
//// import viva_math/autodiff_reverse as ad
////
//// // f(x, y, z) = x² + 2·y·z
//// let tape = ad.empty_tape()
//// let #(x, tape) = ad.input(tape, 1.0)
//// let #(y, tape) = ad.input(tape, 2.0)
//// let #(z, tape) = ad.input(tape, 3.0)
//// let #(x_sq, tape) = ad.mul(tape, x, x)
//// let #(yz, tape) = ad.mul(tape, y, z)
//// let #(yz2, tape) = ad.scale(tape, yz, 2.0)
//// let #(out, tape) = ad.add(tape, x_sq, yz2)
//// let grads = ad.backward(tape, out)
//// // grads[x] = 2x = 2.0
//// // grads[y] = 2z = 6.0
//// // grads[z] = 2y = 4.0
//// ```

import gleam/dict.{type Dict}
import gleam/list

// ============================================================================
// Tape and node representation
// ============================================================================

/// Reference to a node in the tape — an opaque handle.
pub type NodeId =
  Int

/// What kind of operation produced this node. Public because it appears in
/// the body of `Node`, which is reachable from the public `Tape`.
/// Algebraic operation recorded on each tape node. Opaque — callers should
/// build expressions via the high-level forward operations (`add`, `mul`, …)
/// and consume gradients via `backward` + `grad_of`. Direct construction
/// of `Op` values would let callers fabricate inconsistent tapes.
pub opaque type Op {
  Input
  Add(NodeId, NodeId)
  Sub(NodeId, NodeId)
  Mul(NodeId, NodeId)
  Div(NodeId, NodeId)
  Neg(NodeId)
  Scale(NodeId, Float)
  Exp(NodeId)
  Ln(NodeId)
  Sin(NodeId)
  Cos(NodeId)
  Tanh(NodeId)
  Sigmoid(NodeId)
  Pow(NodeId, Float)
}

/// One node on the tape. Public for tape introspection.
/// Single tape node — pairs a forward value with the `Op` that produced it.
/// Opaque (the integrity of the tape depends on this not being constructed
/// outside the module).
pub opaque type Node {
  Node(value: Float, op: Op)
}

/// Computation tape — append-only graph of forward computations.
/// Opaque (callers should treat it as a token threaded through forward ops).
pub opaque type Tape {
  Tape(nodes: Dict(NodeId, Node), next_id: Int)
}

/// Start a new empty computation tape.
pub fn empty_tape() -> Tape {
  Tape(nodes: dict.new(), next_id: 0)
}

fn push(tape: Tape, value: Float, op: Op) -> #(NodeId, Tape) {
  let id = tape.next_id
  let node = Node(value: value, op: op)
  #(id, Tape(nodes: dict.insert(tape.nodes, id, node), next_id: id + 1))
}

/// Register a new input variable on the tape.
pub fn input(tape: Tape, value: Float) -> #(NodeId, Tape) {
  push(tape, value, Input)
}

/// Read the forward value of a node.
pub fn value(tape: Tape, id: NodeId) -> Float {
  case dict.get(tape.nodes, id) {
    Ok(n) -> n.value
    Error(_) -> 0.0
  }
}

// ============================================================================
// Forward operations (record onto tape)
//
// Each operation appends a single node, recording the forward value plus the
// `Op` tag that `backward` will dispatch on for the local derivative.
// ============================================================================

/// `z = a + b`. Local: `∂z/∂a = 1`, `∂z/∂b = 1`.
pub fn add(tape: Tape, a: NodeId, b: NodeId) -> #(NodeId, Tape) {
  push(tape, value(tape, a) +. value(tape, b), Add(a, b))
}

/// `z = a − b`. Local: `∂z/∂a = 1`, `∂z/∂b = −1`.
pub fn sub(tape: Tape, a: NodeId, b: NodeId) -> #(NodeId, Tape) {
  push(tape, value(tape, a) -. value(tape, b), Sub(a, b))
}

/// `z = a · b`. Local: `∂z/∂a = b`, `∂z/∂b = a`.
pub fn mul(tape: Tape, a: NodeId, b: NodeId) -> #(NodeId, Tape) {
  push(tape, value(tape, a) *. value(tape, b), Mul(a, b))
}

/// `z = a / b`. Local: `∂z/∂a = 1/b`, `∂z/∂b = −a/b²`.
pub fn div(tape: Tape, a: NodeId, b: NodeId) -> #(NodeId, Tape) {
  push(tape, value(tape, a) /. value(tape, b), Div(a, b))
}

/// `z = −a`. Local: `∂z/∂a = −1`.
pub fn neg(tape: Tape, a: NodeId) -> #(NodeId, Tape) {
  push(tape, 0.0 -. value(tape, a), Neg(a))
}

/// `z = s · a` with `s` a runtime constant. Local: `∂z/∂a = s`.
pub fn scale(tape: Tape, a: NodeId, s: Float) -> #(NodeId, Tape) {
  push(tape, value(tape, a) *. s, Scale(a, s))
}

/// `z = exp(a)`. Local: `∂z/∂a = exp(a) = z`.
pub fn exp(tape: Tape, a: NodeId) -> #(NodeId, Tape) {
  push(tape, exp_f(value(tape, a)), Exp(a))
}

/// `z = ln(a)`. Local: `∂z/∂a = 1/a`. Caller must ensure `a > 0`.
pub fn ln(tape: Tape, a: NodeId) -> #(NodeId, Tape) {
  push(tape, ln_f(value(tape, a)), Ln(a))
}

/// `z = sin(a)`. Local: `∂z/∂a = cos(a)`.
pub fn sin(tape: Tape, a: NodeId) -> #(NodeId, Tape) {
  push(tape, sin_f(value(tape, a)), Sin(a))
}

/// `z = cos(a)`. Local: `∂z/∂a = −sin(a)`.
pub fn cos(tape: Tape, a: NodeId) -> #(NodeId, Tape) {
  push(tape, cos_f(value(tape, a)), Cos(a))
}

/// `z = tanh(a)`. Local: `∂z/∂a = 1 − tanh²(a) = 1 − z²`.
pub fn tanh(tape: Tape, a: NodeId) -> #(NodeId, Tape) {
  push(tape, tanh_f(value(tape, a)), Tanh(a))
}

/// `z = σ(a)`. Local: `∂z/∂a = σ(a)·(1 − σ(a)) = z·(1 − z)`.
pub fn sigmoid(tape: Tape, a: NodeId) -> #(NodeId, Tape) {
  let v = value(tape, a)
  let s = 1.0 /. { 1.0 +. exp_f(0.0 -. v) }
  push(tape, s, Sigmoid(a))
}

/// `z = aⁿ` (real exponent `n`). Local: `∂z/∂a = n·aⁿ⁻¹`.
pub fn pow(tape: Tape, a: NodeId, n: Float) -> #(NodeId, Tape) {
  push(tape, pow_f(value(tape, a), n), Pow(a, n))
}

// ============================================================================
// Backward pass
// ============================================================================

/// Compute ∂output/∂node for every node, given a scalar output node.
///
/// Returns a Dict mapping each NodeId to its accumulated gradient. The
/// `output` node itself has gradient 1.0.
pub fn backward(tape: Tape, output: NodeId) -> Dict(NodeId, Float) {
  // Initialise gradients with output = 1.0, others = 0.
  let grads = dict.insert(dict.new(), output, 1.0)
  // Walk the tape in reverse insertion order. Since nodes are inserted with
  // monotonically increasing ids, descending from output covers all
  // ancestors.
  walk_back(tape, output, grads)
}

fn walk_back(
  tape: Tape,
  id: NodeId,
  grads: Dict(NodeId, Float),
) -> Dict(NodeId, Float) {
  case id < 0 {
    True -> grads
    False -> {
      case dict.get(tape.nodes, id) {
        Error(_) -> walk_back(tape, id - 1, grads)
        Ok(node) -> {
          let g = case dict.get(grads, id) {
            Ok(v) -> v
            Error(_) -> 0.0
          }
          let grads_after = propagate(tape, node, g, grads)
          walk_back(tape, id - 1, grads_after)
        }
      }
    }
  }
}

fn propagate(
  tape: Tape,
  node: Node,
  g: Float,
  grads: Dict(NodeId, Float),
) -> Dict(NodeId, Float) {
  case node.op {
    Input -> grads
    Add(a, b) -> {
      grads
      |> bump(a, g)
      |> bump(b, g)
    }
    Sub(a, b) -> {
      grads
      |> bump(a, g)
      |> bump(b, 0.0 -. g)
    }
    Mul(a, b) -> {
      let va = value(tape, a)
      let vb = value(tape, b)
      grads
      |> bump(a, g *. vb)
      |> bump(b, g *. va)
    }
    Div(a, b) -> {
      let va = value(tape, a)
      let vb = value(tape, b)
      grads
      |> bump(a, g /. vb)
      |> bump(b, 0.0 -. g *. va /. { vb *. vb })
    }
    Neg(a) -> bump(grads, a, 0.0 -. g)
    Scale(a, s) -> bump(grads, a, g *. s)
    Exp(a) -> bump(grads, a, g *. exp_f(value(tape, a)))
    Ln(a) -> bump(grads, a, g /. value(tape, a))
    Sin(a) -> bump(grads, a, g *. cos_f(value(tape, a)))
    Cos(a) -> bump(grads, a, 0.0 -. g *. sin_f(value(tape, a)))
    Tanh(a) -> {
      let t = tanh_f(value(tape, a))
      bump(grads, a, g *. { 1.0 -. t *. t })
    }
    Sigmoid(a) -> {
      let v = value(tape, a)
      let s = 1.0 /. { 1.0 +. exp_f(0.0 -. v) }
      bump(grads, a, g *. s *. { 1.0 -. s })
    }
    Pow(a, n) -> {
      let va = value(tape, a)
      bump(grads, a, g *. n *. pow_f(va, n -. 1.0))
    }
  }
}

fn bump(
  grads: Dict(NodeId, Float),
  id: NodeId,
  delta: Float,
) -> Dict(NodeId, Float) {
  let current = case dict.get(grads, id) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
  dict.insert(grads, id, current +. delta)
}

/// Extract gradient ∂output/∂input from the backward result.
pub fn grad_of(grads: Dict(NodeId, Float), input: NodeId) -> Float {
  case dict.get(grads, input) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

/// Convenience: gradient of a scalar function with respect to many inputs.
///
/// Runs the function on a fresh tape, performs the backward pass, and
/// returns the gradient list aligned with the input list.
pub fn gradients(
  inputs: List(Float),
  build: fn(Tape, List(NodeId)) -> #(NodeId, Tape),
) -> List(Float) {
  let tape0 = empty_tape()
  let #(ids, tape_with_inputs) = register_inputs(inputs, tape0, [])
  let #(out, final_tape) = build(tape_with_inputs, ids)
  let grads = backward(final_tape, out)
  list.map(ids, fn(id) { grad_of(grads, id) })
}

fn register_inputs(
  values: List(Float),
  tape: Tape,
  acc: List(NodeId),
) -> #(List(NodeId), Tape) {
  case values {
    [] -> #(list.reverse(acc), tape)
    [v, ..rest] -> {
      let #(id, t) = input(tape, v)
      register_inputs(rest, t, [id, ..acc])
    }
  }
}

// ============================================================================
// FFI math helpers
// ============================================================================

@external(erlang, "math", "exp")
fn exp_f(x: Float) -> Float

@external(erlang, "math", "log")
fn ln_f(x: Float) -> Float

@external(erlang, "math", "sin")
fn sin_f(x: Float) -> Float

@external(erlang, "math", "cos")
fn cos_f(x: Float) -> Float

@external(erlang, "math", "tanh")
fn tanh_f(x: Float) -> Float

@external(erlang, "math", "pow")
fn pow_f(x: Float, n: Float) -> Float
