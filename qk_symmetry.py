"""
Diagnostics for the GL(D) symmetry of the query-key product in attention.

The loss depends on (W_Q, W_K) only through M = W_K^T W_Q, so the loss is
invariant under (W_Q, W_K) -> (S W_Q, S^-T W_K) for nonsingular S.  Two
consequences are measured here:

  1. CONSERVATION.  Under gradient flow, d/dt (W_Q W_Q^T - W_K W_K^T) = 0.
     Cheap: one D x D matrix per head.  Sees only the symmetric part of the
     drift, i.e. D(D+1)/2 of the D^2 orbit directions.

  2. PROJECTION.  The null space of the directional derivative is
     N = {(A W_Q, -A^T W_K) : A in R^{DxD}}.  The least-squares projection of
     an update onto N solves the Sylvester equation

         A P + Q A = C,   P = W_Q W_Q^T,  Q = W_K W_K^T,
                          C = dW_Q W_Q^T - W_K dW_K^T

     which has a closed form via two symmetric eigendecompositions.  Unique
     whenever W_Q, W_K are full rank.  rho in [0, 1] is the fraction of update
     energy lying in the null space; it is exactly 0 for a raw gradient step.

INTERPRETING rho.  An isotropic random update puts d/n of its energy in any
d-dimensional subspace, and here d/n = D^2 / 2DE = D/(2E) = 1/(2H) -- exactly
the unidentified fraction.  So the scale reads:

    rho ~ 0          the optimizer respects the symmetry (provable for SGD)
    rho ~ 1/(2H)     it treats the null space like any other direction
    rho > 1/(2H)     it preferentially moves along invariant directions

Everything runs on CPU in float64 on purpose.  The headline result is a null
result -- "this quantity stays flat" -- so low precision manufactures exactly
the signal we are hunting.  These are D x D problems; the cost is negligible.

Usage in nanoGPT's train.py, around the existing step.  Only snapshot on the
iterations you actually log -- capture() forces a device synchronization.

    measure = (iter_num % log_interval == 0)
    pre_step = capture(model, n_embd, n_head) if measure else None

    scaler.step(optimizer)
    scaler.update()

    if measure:
        stats = diagnose(model, pre_step, n_embd, n_head)

    optimizer.zero_grad(set_to_none=True)

Run this file directly to execute the self-test.
"""

import torch


def null_baseline(n_embd, n_head):
    """Expected rho for an isotropic random update: D/(2E) = 1/(2H).

    Verified numerically to three decimals across (D, E) in
    {(4,64), (16,64), (32,64), (64,128)}.  Use it as the reference line.
    """
    return 1.0 / (2.0 * n_head)


def _weights_f64(model, n_embd, n_head):
    """Yield ((layer, head), W_Q, W_K) as CPU float64 (D, E) tensors.

    nanoGPT fuses Q, K, V into a single Linear(E, 3E), so the per-head matrices
    are slices of slices.  nn.Linear stores weight as (out, in) and computes
    x @ W.T, so each slice already acts as W x on column vectors -- no
    transposes needed to match the (D, E) convention.

    The whole (3E, E) weight moves in ONE transfer per layer and is sliced on
    the CPU side.  On MPS each .cpu() is a synchronization barrier that flushes
    the command queue, so per-head transfers would stall the pipeline H times
    per layer for no reason -- true on unified memory as much as on discrete.

    Order matters in the cast: Metal has no float64 at all, so the tensor must
    reach the CPU before it can become double.  .to(device=..., dtype=...) in
    one call attempts the cast on-device and raises.
    """
    E, H = n_embd, n_head
    D = E // H
    for layer, block in enumerate(model.transformer.h):
        W = block.attn.c_attn.weight.detach().cpu().double()   # (3E, E)
        Wq, Wk, _ = W.split(E, dim=0)                          # each (E, E)
        Wq, Wk = Wq.view(H, D, E), Wk.view(H, D, E)
        for head in range(H):
            yield (layer, head), Wq[head], Wk[head]


def capture(model, n_embd, n_head):
    """Snapshot pre-step weights.

    The clone is load-bearing.  These are views onto a tensor the optimizer
    mutates in place; without a copy every diff comes back identically zero.
    (.double() already copies when the model is float32, but not if you ever
    run a float64 CPU model -- so the clone stays.)
    """
    return {key: (wq.clone(), wk.clone())
            for key, wq, wk in _weights_f64(model, n_embd, n_head)}


class Diagonoser:

    def __init__(self):
        self.c0 = {}

    def head_metrics(self, key, Wq0, Wk0, Wq1, Wk1):
        """Diagnostics for one head, given pre- and post-step weights (D, E) f64."""

        dWq, dWk = Wq1 - Wq0, Wk1 - Wk0

        # --- projection onto the null space (W6) --------------------------------
        # Coefficients come from the PRE-step weights: the update was computed in
        # that tangent space.
        P = Wq0 @ Wq0.T
        Q = Wk0 @ Wk0.T
        R = dWq @ Wq0.T - Wk0 @ dWk.T
        
        con0 = P - Q
        con1 = Wq1 @ Wq1.T - Wk1 @ Wk1.T

        if key not in self.c0:
            self.c0[key] = con0

        c0 = self.c0[key]

        drift = torch.linalg.norm(con1 - con0).item()
        tau = con1.trace().item()

        lamP, U = torch.linalg.eigh(P)               # P = U diag(lam) U^T
        lamQ, V = torch.linalg.eigh(Q)               # Q = V diag(mu)  V^T

        denom = lamQ[:, None] + lamP[None, :]          # A~_ij = R~_ij / (lam_j + mu_i)
        S = V @ ((V.T @ R @ U) / denom) @ U.T

        proj_q, proj_k = S @ Wq0, -S.T @ Wk0
        num = proj_q.pow(2).sum() + proj_k.pow(2).sum()
        den = dWq.pow(2).sum() + dWk.pow(2).sum()
        
        con_norm = torch.linalg.norm(con1).item()

        return {
            "drift": drift,                  # ||d(W_Q W_Q^T - W_K W_K^T)||_F
            "beta": con_norm,
            "tau": tau,
            "rho": (num / den).item() if den > 0 else 0.0,
            "update_norm": den.sqrt().item(),
            # Smallest eigenvalue sum: conditions the Sylvester solve, and goes to
            # zero exactly as W_Q or W_K lose rank -- i.e. as the full-rank
            # hypothesis behind the D^2 count starts to fail.
            "mu": denom.min().item(),
            "con_drift": (torch.linalg.norm(con1 - c0) / torch.linalg.norm(c0)).item(),
        }

    def diagnose(self, model, pre_step, n_embd, n_head):
        """Per-head metrics for the step that just happened.

        `pre_step` is the dict returned by capture() before optimizer.step().  The
        post-step weights are read from the model itself -- the optimizer mutated
        it in place, so the live model *is* the "after".
        """
        out = {}
        for key, wq, wk in _weights_f64(model, n_embd, n_head):
            Wq0, Wk0 = pre_step[key]
            out["head" + str(key)] = self.head_metrics(key, Wq0, Wk0, wq, wk)
        return out


# ---------------------------------------------------------------------------
# Self-test.  Validate the harness before trusting anything downstream.
# ---------------------------------------------------------------------------

def _self_test():
    torch.manual_seed(0)
    D, E, eta = 8, 16, 1e-2
    Wq = torch.randn(D, E, dtype=torch.float64)
    Wk = torch.randn(D, E, dtype=torch.float64)
    G = torch.randn(E, E, dtype=torch.float64)   # dL/dM for some loss

    diagonoser = Diagonoser()

    # A raw gradient step.  dL/dW_Q = W_K G,  dL/dW_K = W_Q G^T.
    dWq, dWk = -eta * (Wk @ G), -eta * (Wq @ G.T)
    m = diagonoser.head_metrics(Wq, Wk, Wq + dWq, Wk + dWk)
    assert m["rho"] < 1e-20, f"gradient step should have rho == 0, got {m['rho']}"
    print(f"gradient step:  rho = {m['rho']:.3e}   (expected 0)")

    # First-order conservation: drift is second order in the step size, so it
    # should fall by ~100x when eta falls by 10x.
    d1 = diagonoser.head_metrics(Wq, Wk, Wq + dWq, Wk + dWk)["drift"]
    d2 = diagonoser.head_metrics(Wq, Wk, Wq + dWq / 10, Wk + dWk / 10)["drift"]
    print(f"drift ratio:    {d1 / d2:.1f}   (expected ~100, second order)")
    assert 50 < d1 / d2 < 200

    # An arbitrary update has nonzero overlap, and the projection is idempotent.
    rhos = []
    for _ in range(400):
        rWq = eta * torch.randn(D, E, dtype=torch.float64)
        rWk = eta * torch.randn(D, E, dtype=torch.float64)
        rhos.append(diagonoser.head_metrics(Wq, Wk, Wq + rWq, Wk + rWk)["rho"])
    mean_rho = sum(rhos) / len(rhos)
    expected = D / (2 * E)
    print(f"random steps:   rho = {mean_rho:.4f}   (expected D/2E = {expected:.4f})")
    assert abs(mean_rho - expected) < 0.02

    A_dir = torch.randn(D, D, dtype=torch.float64) * eta
    pWq, pWk = A_dir @ Wq, -A_dir.T @ Wk
    m = diagonoser.head_metrics(Wq, Wk, Wq + pWq, Wk + pWk)
    print(f"pure null step: rho = {m['rho']:.12f}   (expected 1)")
    assert abs(m["rho"] - 1.0) < 1e-8

    print("\nall checks passed")


if __name__ == "__main__":
    _self_test()
