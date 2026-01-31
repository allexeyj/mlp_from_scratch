"""
Numerical gradient check (finite differences) for the NumPy implementation.

This is one of the highest-value educational tools:
- verifies backward() correctness
- helps debug sign/shape mistakes
- builds intuition that backprop is "just chain rule + bookkeeping"
"""

import numpy as np

from numpy_impl import MLP, CrossEntropyLoss


def numerical_grad(model: MLP, criterion: CrossEntropyLoss, X: np.ndarray, y: np.ndarray, eps: float = 1e-5):
    """
    Compute numerical gradients for all parameters via central differences.
    Returns a dict param_name -> grad_array (same shape as param).
    """
    num_grads = {}
    params = model.get_params()

    for name, param in params.items():
        g = np.zeros_like(param)
        it = np.ndindex(param.shape)

        for idx in it:
            old = param[idx]

            param[idx] = old + eps
            loss_pos = criterion(model.forward(X), y)

            param[idx] = old - eps
            loss_neg = criterion(model.forward(X), y)

            param[idx] = old
            g[idx] = (loss_pos - loss_neg) / (2 * eps)

        num_grads[name] = g

    return num_grads


def relative_error(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> float:
    denom = np.maximum(tol, np.abs(a) + np.abs(b))
    return np.max(np.abs(a - b) / denom)


def main():
    np.random.seed(0)

    # Small problem to keep it fast
    batch = 4
    input_dim = 5
    hidden_dims = [4]
    output_dim = 3

    X = np.random.randn(batch, input_dim).astype(np.float64)
    y = np.array([0, 1, 2, 1], dtype=np.int64)

    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout=0.0)
    criterion = CrossEntropyLoss()

    # Analytical grads
    logits = model.forward(X)
    loss = criterion(logits, y)
    dout = criterion.backward()
    model.backward(dout)

    grads = model.get_grads()

    # Numerical grads
    num_grads = numerical_grad(model, criterion, X, y, eps=1e-5)

    # Report
    print("=" * 70)
    print("Gradient check (finite differences)")
    print("=" * 70)
    print(f"Loss: {loss:.6f}\n")

    all_ok = True
    for name in grads:
        g_analytical = grads[name].astype(np.float64)
        g_numerical = num_grads[name].astype(np.float64)

        max_abs = float(np.max(np.abs(g_analytical - g_numerical)))
        rel = float(relative_error(g_analytical, g_numerical))

        # Rules of thumb (not strict):
        # - rel error ~1e-6..1e-7 is excellent
        # - ~1e-4 can still be acceptable depending on eps / non-smooth points (ReLU)
        ok = rel < 1e-4 or max_abs < 1e-6
        all_ok = all_ok and ok

        print(f"{name:>12s} | max_abs_diff={max_abs:.3e} | rel_error={rel:.3e} | {'OK' if ok else 'CHECK'}")

    print("\nResult:", "✅ PASSED" if all_ok else "❌ NEEDS ATTENTION")
    print(
        "\nNote: ReLU is non-smooth at 0, so tiny discrepancies are possible "
        "if some activations are exactly near 0. Re-run with a different seed if needed."
    )


if __name__ == "__main__":
    main()
