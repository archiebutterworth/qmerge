# =============================================================================
#  SSO FRAMEWORK -- earlier standalone copy
# =============================================================================
#
#  Functionally the same stack as sso_prep.py, extracted at an earlier point.
#  Both are kept because different notebooks import different ones:
#
#      sso_prep.py            <- 01_qmerge_engine.ipynb, sso_rowprep.py
#      sso_framework_setup.py <- 02_haar_scaling_sweep.ipynb,
#                                supplementary/SU4 Reweighting POC.ipynb,
#                                supplementary/IPR Reduction Ansatz Comparison.ipynb
#
#  If you are reading the method rather than running a notebook, read sso_prep.py
#  -- it is the maintained one and carries the fuller header.
#
#  Importing this module prints its self-check output. That is expected.
#
# =============================================================================

# =====================================================================
#  SSO framework -- full setup in one cell.
#  Paste into a single Jupyter cell and run once; defines everything the
#  encoding notebooks use (warm-start + cold SSO drivers, image<->MPS,
#  chi_fidelity, replay/reconstruction helpers).
#  Needs (phys env): numpy, torch, scipy, quimb, qiskit, mps_to_circuit.
#  Verbatim copy of `archie_test_ultrasound copy.ipynb` setup cells.
# =====================================================================

# --- Definition cell (from new.ipynb cell 2) ---
import numpy as np
import torch
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor import PTensor
from functools import partial
import math

qtn.set_contract_strategy("auto-hq")

# complex128 everywhere so identity checks are numerically sharp
torch.set_default_dtype(torch.float64)
RNG = np.random.default_rng(0)
torch.manual_seed(0)

# --- Definition cell (from new.ipynb cell 4) ---
def su2_matrix(params: torch.Tensor) -> torch.Tensor:
    """3 real params -> SU(2) unitary, reshaped as a 2x2 tensor."""
    device = params.device
    dtype = torch.complex128
    sx = torch.tensor([[0, 1],  [1, 0]],   dtype=dtype, device=device)
    sy = torch.tensor([[0,-1j], [1j, 0]],  dtype=dtype, device=device)
    sz = torch.tensor([[1, 0],  [0,-1]],   dtype=dtype, device=device)
    Theta = params[0]*sx + params[1]*sy + params[2]*sz
    U = torch.linalg.matrix_exp(1j * Theta)
    return U.reshape(2, 2)


def su4_matrix(params: torch.Tensor) -> torch.Tensor:
    """15 real params -> SU(4) unitary, reshaped as a rank-4 tensor (2,2,2,2)."""
    N = 4
    device = params.device
    dtype = torch.complex128

    Theta = torch.zeros((N, N), dtype=dtype, device=device)

    # Diagonal generators (Cartan subalgebra)
    for k in range(N - 1):
        vec = torch.zeros(N, dtype=torch.float64, device=device)
        vec[: k + 1] = 1.0
        vec[k + 1] = -(k + 1)
        Theta = Theta + params[k] * torch.diag(vec).to(dtype)

    # Off-diagonal generators (real-symmetric + imag-antisymmetric)
    idx = N - 1
    for i in range(N):
        for j in range(i + 1, N):
            mr = torch.zeros((N, N), dtype=dtype, device=device)
            mi = torch.zeros((N, N), dtype=dtype, device=device)
            mr[i, j] = mr[j, i] = 1.0
            mi[i, j] = 1.0j
            mi[j, i] = -1.0j
            Theta = Theta + params[idx] * mr + params[idx + 1] * mi
            idx += 2

    U = torch.linalg.matrix_exp(1j * Theta)
    return U.reshape(2, 2, 2, 2)


def su8_matrix(params: torch.Tensor) -> torch.Tensor:
    """63 real params -> SU(8) unitary, reshaped as a rank-6 tensor (2,2,2,2,2,2)."""
    N = 8
    device = params.device
    dtype = torch.complex128

    Theta = torch.zeros((N, N), dtype=dtype, device=device)

    for k in range(N - 1):
        vec = torch.zeros(N, dtype=torch.float64, device=device)
        vec[: k + 1] = 1.0
        vec[k + 1] = -(k + 1)
        Theta = Theta + params[k] * torch.diag(vec).to(dtype)

    idx = N - 1
    for i in range(N):
        for j in range(i + 1, N):
            mr = torch.zeros((N, N), dtype=dtype, device=device)
            mi = torch.zeros((N, N), dtype=dtype, device=device)
            mr[i, j] = mr[j, i] = 1.0
            mi[i, j] = 1.0j
            mi[j, i] = -1.0j
            Theta = Theta + params[idx] * mr + params[idx + 1] * mi
            idx += 2

    U = torch.linalg.matrix_exp(1j * Theta)
    return U.reshape(2, 2, 2, 2, 2, 2)


# Sanity: zero params must give exact identity (this is what the original
# notebook paid a whole optimisation to approximate).
def _check_identity_from_zero():
    for nparams, fn, dim in [(3, su2_matrix, 2), (15, su4_matrix, 4), (63, su8_matrix, 8)]:
        M = fn(torch.zeros(nparams, dtype=torch.float64)).reshape(dim, dim)
        err = torch.linalg.norm(M - torch.eye(dim, dtype=torch.complex128)).item()
        assert err < 1e-12, f"{fn.__name__}(0) != I (err={err})"
    print("OK: su2_matrix(0) == I, su4_matrix(0) == I, su8_matrix(0) == I.")


_check_identity_from_zero()

# --- Definition cell (from new.ipynb cell 6) ---
def make_hybrid_staircase(n: int, eps: float = 0.0, seed: int | None = None) -> qtn.TensorNetwork:
    """
    Build the SU4 -> SU8 x (n-3) -> SU4 -> SU2 staircase for an n-qubit system.

    Parameters
    ----------
    n : int
        Number of qubits (>= 4 so that we have at least one middle SU8 block).
    eps : float
        Std-dev of the Gaussian perturbation applied to the zero initialisation.
        eps = 0.0  -> the staircase is exactly the identity map k{i} -> b{i}.
        eps > 0    -> gives a small random perturbation so the optimiser can
                      see a non-zero gradient.
    seed : int | None
        RNG seed for the perturbation. ``None`` means use torch's default stream.
    """
    assert n >= 4, "Need n >= 4 for the SU4 -> SU8 ... -> SU4 -> SU2 staircase."

    if seed is not None:
        torch.manual_seed(seed)

    def _init(nparams):
        p = torch.zeros(nparams, dtype=torch.float64)
        if eps > 0:
            p = p + eps * torch.randn(nparams, dtype=torch.float64)
        p.requires_grad_(True)
        return p

    tensors = []
    k_free = 0
    b_free = 0
    c_counter = 0
    n_mid = n - 3  # number of SU8 blocks in the middle

    # 1) Leading SU4 on (k0, k1) -> (b0, c0)
    in_inds  = [f"k{k_free}", f"k{k_free + 1}"]; k_free += 2
    out_inds = [f"b{b_free}", f"c{c_counter}"]; b_free += 1; c_counter += 1
    tensors.append(PTensor(su4_matrix, _init(15),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU4", "L0"}))
    prev_wire = out_inds[1]

    # 2) First SU8: in = (prev_wire, k2, k3) -> out = (b1, c1, c2)
    in_inds  = [prev_wire, f"k{k_free}", f"k{k_free + 1}"]; k_free += 2
    out_inds = [f"b{b_free}", f"c{c_counter}", f"c{c_counter + 1}"]
    b_free += 1; c_counter += 2
    tensors.append(PTensor(su8_matrix, _init(63),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU8", "L1"}))
    prev_top = out_inds[1:]

    # 3) Remaining SU8 blocks
    for i in range(1, n_mid):
        in_inds  = [prev_top[0], prev_top[1], f"k{k_free}"]; k_free += 1
        out_inds = [f"b{b_free}", f"c{c_counter}", f"c{c_counter + 1}"]
        b_free += 1; c_counter += 2
        tensors.append(PTensor(su8_matrix, _init(63),
                               inds=tuple(in_inds + out_inds),
                               tags={"SU8", f"L{i + 1}"}))
        prev_top = out_inds[1:]

    # 4) Trailing SU4: in = (prev_top[0], prev_top[1]) -> out = (b_{n-2}, c_last)
    in_inds  = [prev_top[0], prev_top[1]]
    out_inds = [f"b{b_free}", f"c{c_counter}"]; b_free += 1; c_counter += 1
    tensors.append(PTensor(su4_matrix, _init(15),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU4", f"L{n_mid + 1}"}))
    prev_wire = out_inds[1]

    # 5) Trailing SU2: in = prev_wire -> b_{n-1}
    in_inds  = [prev_wire]
    out_inds = [f"b{b_free}"]; b_free += 1
    tensors.append(PTensor(su2_matrix, _init(3),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU2", f"L{n_mid + 2}"}))

    tn = qtn.TensorNetwork(tensors)
    return tn

# --- Definition cell (from new.ipynb cell 12) ---
def _swap_kb(tn: qtn.TensorNetwork) -> None:
    """In-place: relabel `k{i}` <-> `b{i}` on the outer indices of `tn`."""
    k_inds = sorted((ix for ix in tn.outer_inds() if ix.startswith("k")),
                    key=lambda s: int(s[1:]))
    b_inds = sorted((ix for ix in tn.outer_inds() if ix.startswith("b")),
                    key=lambda s: int(s[1:]))
    tmp = [f"__tmp{i}" for i in range(len(k_inds))]
    tn.reindex_({k_inds[i]: tmp[i]    for i in range(len(k_inds))})
    tn.reindex_({b_inds[i]: k_inds[i] for i in range(len(b_inds))})
    tn.reindex_({tmp[i]:    b_inds[i] for i in range(len(tmp))})


def dense_state_from_op_mps(op: qtn.TensorNetwork,
                            mps: qtn.MatrixProductState) -> torch.Tensor:
    """
    Contract `op | mps` to a dense state tensor of shape ``(2,)*n`` indexed
    by ``(b0, b1, ..., b_{n-1})``.

    This is the gradient-clean contraction: it uses only index relabelling
    and dense contractions (no QR, no truncation), so autograd flows
    through every operation. For the circuit sizes in this notebook
    (``n <= 10``), a dense state is ~2^n complex128 numbers, which is
    cheap (2^10 = 1024 numbers).
    """
    op = op.copy()
    mps = mps.copy()
    n = mps.L

    _swap_kb(op)

    # Force mps tensors to torch complex128 so the "torch" contraction backend
    # has a homogeneous view of the combined TN. (`op` tensors are already
    # torch because their PTensor parameters were built as torch tensors.)
    def _to_torch_complex(x):
        if isinstance(x, torch.Tensor):
            return x.to(torch.complex128) if x.dtype != torch.complex128 else x
        return torch.as_tensor(np.asarray(x), dtype=torch.complex128)

    mps.apply_to_arrays(_to_torch_complex)

    combined = op | mps

    b_inds = tuple(f"b{i}" for i in range(n))
    T = combined.contract(all, output_inds=b_inds,
                          optimize="auto-hq", backend="torch")
    data = T.data
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)
    return data


def contract_layer_to_mps(op: qtn.TensorNetwork,
                          mps: qtn.MatrixProductState,
                          max_bond: int | None = None) -> qtn.MatrixProductState:
    """
    Produce the evolved state as a proper (detached) left-canonical MPS
    with torch complex128 tensors.

    Goes through `dense_state_from_op_mps` and then reshapes / MPS-decomposes
    the result. Detached from autograd (the evolved state is a fact about the
    optimised layer; subsequent layers should not back-prop through it).
    """
    psi = dense_state_from_op_mps(op, mps).detach()
    psi_np = psi.cpu().numpy()
    n = mps.L
    new = qtn.MatrixProductState.from_dense(psi_np.reshape(*([2] * n)),
                                            dims=[2] * n,
                                            site_ind_id="k{}",
                                            site_tag_id="I{}")
    if max_bond is not None:
        new.compress(max_bond=max_bond)
    new.left_canonicalize(inplace=True)
    # Convert MPS tensors to torch so downstream contractions with other
    # torch-backed TNs are homogeneous (mixing numpy + torch tensors in one
    # contract call triggers backend-dispatch errors in autoray/cotengra).
    new.apply_to_arrays(lambda x: torch.as_tensor(np.asarray(x),
                                                  dtype=torch.complex128))
    return new

# --- Definition cell (from new.ipynb cell 14) ---
def _schmidt_sq_at_cut(psi_nd: torch.Tensor, cut: int) -> torch.Tensor:
    """
    Schmidt-squared spectrum for the bipartition ``[0..cut] | [cut+1..n-1]``
    of a dense state tensor of shape ``(2,)*n``.

    Returns the squared singular values of the reshaped matrix, already
    normalised so that they sum to ``<psi|psi>``.
    """
    n = psi_nd.ndim
    left_dim  = 2 ** (cut + 1)
    right_dim = 2 ** (n - cut - 1)
    M = psi_nd.reshape(left_dim, right_dim)
    s = torch.linalg.svdvals(M)
    return (s * s).to(torch.float64)


def tail_mass_loss(op: qtn.TensorNetwork,
                   init_tn: qtn.MatrixProductState,
                   g: float = 0.0,
                   topk: int = 4, #Change for different loss functions
                   eps: float = 1e-12) -> torch.Tensor:
    """
    Sum over all ``n-1`` bipartite cuts of ``T_j * (1 + g * H_tail_norm)``.

    The state ``|psi> = op |init_tn>`` is computed densely, its Schmidt
    spectrum at every cut is read off via ``torch.linalg.svdvals``
    (which is differentiable), and the tail mass at each cut is summed.
    When ``g = 0`` this is just the total tail mass - a chi=k infidelity
    proxy.
    """
    psi = dense_state_from_op_mps(op, init_tn)
    n = psi.ndim
    # Normalise psi defensively; if op is unitary and init_tn is normalised,
    # this is a no-op (up to floating point).
    norm_sq = (psi.conj() * psi).sum().real
    psi = psi / torch.sqrt(torch.clamp(norm_sq, min=eps))

    vals = []
    for cut in range(n - 1):
        lam2 = _schmidt_sq_at_cut(psi, cut)  # already normalised (sum ~ 1)
        lam2 = torch.clamp(lam2, min=0.0)
        p_sorted, _ = torch.sort(lam2, descending=True)

        d = p_sorted.numel()
        if d <= topk:
            vals.append(torch.tensor(0.0, dtype=torch.float64, device=psi.device))
            continue

        Tj = 1.0 - p_sorted[:topk].sum()
        if g == 0.0:
            vals.append(Tj)
            continue

        tail = p_sorted[topk:]
        d_tail = tail.numel()
        if d_tail <= 1 or Tj <= eps:
            vals.append(Tj)
            continue
        r = torch.clamp(tail / torch.clamp(Tj, min=eps), min=eps)
        H_tail = -(r * (r.log() / math.log(2.0))).sum()
        H_tail_norm = H_tail / math.log2(d_tail)
        vals.append(Tj * (1.0 + g * H_tail_norm))

    if not vals:
        return torch.tensor(0.0, dtype=torch.float64)
    return torch.stack(vals).sum()

# --- Definition cell (from new.ipynb cell 16) ---
def chi_fidelity(mps: qtn.MatrixProductState, chi: int) -> float:
    """|<compressed_chi | original>|^2 after both are normalised."""
    m = mps.copy()
    m.apply_to_arrays(lambda x: x.detach() if isinstance(x, torch.Tensor) else x)
    m_np = m.copy()
    m_np.apply_to_arrays(lambda x: x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x))
    x = m_np.copy()
    x.compress(max_bond=chi)
    x /= x.norm()
    ref = m_np / m_np.norm()
    return float(abs((x.conj() | ref).contract(all)) ** 2)


def assert_layer_unitary(tn: qtn.TensorNetwork, atol: float = 1e-5) -> None:
    """Every staircase gate should still be a unitary after optimisation."""
    for t in tn.tensors:
        if len(t.inds) == 4:
            M = t.data.reshape(4, 4)
        elif len(t.inds) == 6:
            M = t.data.reshape(8, 8)
        elif len(t.inds) == 2:
            M = t.data.reshape(2, 2)
        else:
            raise ValueError(f"Unexpected tensor rank {len(t.inds)}.")
        if isinstance(M, torch.Tensor):
            M = M.detach().cpu().numpy()
        I = np.eye(M.shape[0])
        err = np.linalg.norm(M.conj().T @ M - I)
        assert err < atol, f"Gate is not unitary (err = {err:.3e})"

# --- Definition cell (from new.ipynb cell 18) ---
def _to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def optimise_one_layer(mps: qtn.MatrixProductState,
                       loss_fn,
                       n_iter: int = 300,
                       tol: float = 1e-14,
                       eps_init: float = 1e-3,
                       seed: int | None = None,
                       verbose: bool = False):
    """
    Optimise a fresh hybrid staircase to minimise `loss_fn(layer, mps)`.

    - Guarantees the returned layer's loss <= loss of the identity staircase.
    - Returns the *best-loss* iterate seen during optimisation, not the last.

    `loss_fn` must be callable as `loss_fn(op, init_tn=mps)` and return a
    scalar torch tensor (this is satisfied by `tail_mass_loss` and any
    `functools.partial` over its `g`/`topk` keyword arguments).
    """
    n = mps.L
    layer = make_hybrid_staircase(n, eps=eps_init, seed=seed)

    # Identity layer and its loss - this is our absolute lower bound on output quality.
    id_layer = make_hybrid_staircase(n, eps=0.0)
    with torch.no_grad():
        id_loss = float(loss_fn(id_layer, init_tn=mps))
        init_loss = float(loss_fn(layer, init_tn=mps))
    if verbose:
        print(f"  identity loss = {id_loss:.6e}   "
              f"perturbed init loss = {init_loss:.6e}")

    # Best-loss tracker. We wrap the loss so that every forward evaluation
    # (including those L-BFGS-B does for line searches) has a chance to
    # update the snapshot.
    best = {"val": float("inf"), "params": None}

    def _tracked(op, init_tn):
        val = loss_fn(op, init_tn=init_tn)
        scalar = float(val.detach()) if isinstance(val, torch.Tensor) else float(val)
        if scalar < best["val"]:
            params = op.get_params()
            best["val"] = scalar
            best["params"] = {k: _to_numpy(v).copy() for k, v in params.items()}
        return val

    tnopt = qtn.TNOptimizer(
        layer,
        _tracked,
        loss_constants={"init_tn": mps},
        optimizer="l-bfgs-b",
        autodiff_backend="torch",
        progbar=verbose,
    )
    _ = tnopt.optimize(n=n_iter, tol=tol)

    # Restore the best-loss iterate
    if best["params"] is not None:
        layer_best = layer.copy()
        layer_best.set_params({k: torch.as_tensor(v) for k, v in best["params"].items()})
        best_loss = best["val"]
    else:
        layer_best = layer
        with torch.no_grad():
            best_loss = float(loss_fn(layer_best, init_tn=mps))

    # Make sure layer tensors are torch so downstream contractions work.
    layer_best.apply_to_arrays(lambda x: torch.as_tensor(x))

    # ----- Identity fallback. ------------------------------------------------
    # If the optimiser never beat the identity, use the identity instead.
    if best_loss > id_loss - 1e-12:
        if verbose:
            print(f"  [fallback] optimised loss {best_loss:.6e} >= identity "
                  f"loss {id_loss:.6e}; returning identity layer.")
        id_layer.apply_to_arrays(lambda x: torch.as_tensor(x))
        return id_layer, id_loss

    assert_layer_unitary(layer_best)
    return layer_best, best_loss

# --- Definition cell (from new.ipynb cell 20) ---
def sequential_disentangle(target_mps: qtn.MatrixProductState,
                           num_layers: int,
                           g: float = 0.0,
                           topk: int = 2,
                           n_iter: int = 300,
                           eps_init: float = 1e-3,
                           mono_tol: float = 1e-8,
                           verbose: bool = True) -> dict:
    """
    Run `num_layers` staircases and return:
      {'mps_states': [mps after each layer],
       'params':     [layer params],
       'layers':     [the layer TNs],
       'chi2_fids':  [chi=2 fidelity at each stage, including stage 0 = input],
       'chi4_fids':  [chi=4 fidelity at each stage, including stage 0 = input]}
    """
    mps = target_mps.copy()
    mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))

    loss = partial(tail_mass_loss, g=g, topk=topk)

    chi2_fids = [chi_fidelity(mps, 2)]
    chi4_fids = [chi_fidelity(mps, 4)]
    layers = []
    params = []
    mps_states = [mps.copy()]

    if verbose:
        print(f"[init]   chi=2 fid = {chi2_fids[0]:.6f}   "
              f"chi=4 fid = {chi4_fids[0]:.6f}   "
              f"max_bond = {mps.max_bond()}")

    for L in range(num_layers):
        if verbose:
            print(f"\n---- layer {L} ----")
        layer, layer_loss = optimise_one_layer(
            mps, loss, n_iter=n_iter, eps_init=eps_init, seed=L, verbose=verbose,
        )
        layers.append(layer)
        params.append(layer.get_params())

        new_mps = contract_layer_to_mps(layer, mps)

        # Detach torch graph to prevent memory blow-up across layers.
        for t in new_mps.tensors:
            t.modify(data=t.data.detach().clone())

        # Renormalise (contract_layer_to_mps returns a left-canonical MPS,
        # which should already be normalised to 1, but enforce defensively).
        new_mps /= new_mps.norm()

        f2 = chi_fidelity(new_mps, 2)
        f4 = chi_fidelity(new_mps, 4)

        # ---- Monotonicity guarantee ----------------------------------------
        # assert f2 >= chi2_fids[-1] - mono_tol, (
        #     f"chi=2 fidelity DROPPED at layer {L}: "
        #     f"{chi2_fids[-1]:.8f} -> {f2:.8f}"
        # )
        # assert f4 >= chi4_fids[-1] - mono_tol, (
        #     f"chi=4 fidelity DROPPED at layer {L}: "
        #     f"{chi4_fids[-1]:.8f} -> {f4:.8f}"
        # )

        chi2_fids.append(f2)
        chi4_fids.append(f4)
        mps = new_mps
        mps_states.append(mps.copy())

        if verbose:
            print(f"[layer {L}] chi=2 fid = {f2:.6f}   chi=4 fid = {f4:.6f}   "
                  f"max_bond = {mps.max_bond()}   loss = {layer_loss:.4e}")

    return {
        "mps_states": mps_states,
        "params": params,
        "layers": layers,
        "chi2_fids": chi2_fids,
        "chi4_fids": chi4_fids,
    }

# --- Definition cell (from new.ipynb cell 37) ---
def make_chi2_staircase(n: int, eps: float = 0.0, seed: int | None = None) -> qtn.TensorNetwork:
    """
    Build the chi=2 staircase: SU(4) -> SU(4) x (n-2) -> SU(2) for n qubits.

    Each SU(4) carries one chi=2 wire along the chain, so the produced state
    has maximum bipartite Schmidt rank 2. This is the chi=2-compressed
    analogue of `make_hybrid_staircase` (which carries chi=4 via SU(8)s).

    Parameters
    ----------
    n : int
        Number of qubits (>= 2).
    eps : float
        Std-dev of the Gaussian perturbation applied to the zero initialisation.
        eps = 0.0  -> the staircase is exactly the identity map k{i} -> b{i}.
        eps > 0    -> small random perturbation so the optimiser sees gradient.
    seed : int | None
        RNG seed for the perturbation. None means use torch's default stream.
    """
    assert n >= 2, "Need n >= 2 for the SU4 -> SU4 ... -> SU2 staircase."

    if seed is not None:
        torch.manual_seed(seed)

    def _init(nparams):
        p = torch.zeros(nparams, dtype=torch.float64)
        if eps > 0:
            p = p + eps * torch.randn(nparams, dtype=torch.float64)
        p.requires_grad_(True)
        return p

    tensors = []
    k_free = 0
    b_free = 0
    c_counter = 0
    n_mid = n - 2  # number of middle SU4 blocks

    # 1) Leading SU4 on (k0, k1) -> (b0, c0)
    in_inds  = [f"k{k_free}", f"k{k_free + 1}"]; k_free += 2
    out_inds = [f"b{b_free}", f"c{c_counter}"]; b_free += 1; c_counter += 1
    tensors.append(PTensor(su4_matrix, _init(15),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU4", "L0"}))
    prev_wire = out_inds[1]

    # 2) Middle SU4 blocks: (prev_wire, k_new) -> (b_new, c_new)
    for i in range(n_mid):
        in_inds  = [prev_wire, f"k{k_free}"]; k_free += 1
        out_inds = [f"b{b_free}", f"c{c_counter}"]; b_free += 1; c_counter += 1
        tensors.append(PTensor(su4_matrix, _init(15),
                               inds=tuple(in_inds + out_inds),
                               tags={"SU4", f"L{i + 1}"}))
        prev_wire = out_inds[1]

    # 3) Trailing SU2: (prev_wire) -> b_{n-1}
    in_inds  = [prev_wire]
    out_inds = [f"b{b_free}"]; b_free += 1
    tensors.append(PTensor(su2_matrix, _init(3),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU2", f"L{n_mid + 1}"}))

    tn = qtn.TensorNetwork(tensors)
    return tn

# --- Definition cell (from new.ipynb cell 45) ---
def optimise_one_layer_chi2(mps: qtn.MatrixProductState,
                            loss_fn,
                            n_iter: int = 300,
                            tol: float = 1e-14,
                            eps_init: float = 1e-3,
                            seed: int | None = None,
                            verbose: bool = False):
    """
    Optimise a fresh chi=2 staircase to minimise ``loss_fn(layer, mps)``.

    Same contract as ``optimise_one_layer``:
      - Returns the *best-loss* iterate seen during optimisation.
      - Falls back to the identity layer if the optimiser never beats it.
      - Returns ``(layer, best_loss)``.
    """
    n = mps.L
    layer = make_chi2_staircase(n, eps=eps_init, seed=seed)

    id_layer = make_chi2_staircase(n, eps=0.0)
    with torch.no_grad():
        id_loss   = float(loss_fn(id_layer, init_tn=mps))
        init_loss = float(loss_fn(layer,    init_tn=mps))
    if verbose:
        print(f"  identity loss = {id_loss:.6e}   "
              f"perturbed init loss = {init_loss:.6e}")

    best = {"val": float("inf"), "params": None}

    def _tracked(op, init_tn):
        val = loss_fn(op, init_tn=init_tn)
        scalar = float(val.detach()) if isinstance(val, torch.Tensor) else float(val)
        if scalar < best["val"]:
            params = op.get_params()
            best["val"] = scalar
            best["params"] = {k: _to_numpy(v).copy() for k, v in params.items()}
        return val

    tnopt = qtn.TNOptimizer(
        layer,
        _tracked,
        loss_constants={"init_tn": mps},
        optimizer="l-bfgs-b",
        autodiff_backend="torch",
        progbar=verbose,
    )
    _ = tnopt.optimize(n=n_iter, tol=tol)

    if best["params"] is not None:
        layer_best = layer.copy()
        layer_best.set_params({k: torch.as_tensor(v) for k, v in best["params"].items()})
        best_loss = best["val"]
    else:
        layer_best = layer
        with torch.no_grad():
            best_loss = float(loss_fn(layer_best, init_tn=mps))

    layer_best.apply_to_arrays(lambda x: torch.as_tensor(x))

    if best_loss > id_loss - 1e-12:
        if verbose:
            print(f"  [fallback] optimised loss {best_loss:.6e} >= identity "
                  f"loss {id_loss:.6e}; returning identity layer.")
        id_layer.apply_to_arrays(lambda x: torch.as_tensor(x))
        return id_layer, id_loss

    assert_layer_unitary(layer_best)
    return layer_best, best_loss

# --- Definition cell (from new.ipynb cell 47) ---
def sequential_disentangle_chi2(target_mps: qtn.MatrixProductState,
                                num_layers: int,
                                g: float = 0.0,
                                topk: int = 2,
                                n_iter: int = 300,
                                eps_init: float = 1e-3,
                                mono_tol: float = 1e-8,
                                verbose: bool = True) -> dict:
    """
    Run ``num_layers`` chi=2 staircases sequentially. Returns the same dict
    shape as ``sequential_disentangle``:

        {'mps_states', 'params', 'layers', 'chi2_fids', 'chi4_fids'}
    """
    mps = target_mps.copy()
    mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))

    loss = partial(tail_mass_loss, g=g, topk=topk)

    chi2_fids = [chi_fidelity(mps, 2)]
    chi4_fids = [chi_fidelity(mps, 4)]
    layers = []
    params = []
    mps_states = [mps.copy()]

    if verbose:
        print(f"[init]   chi=2 fid = {chi2_fids[0]:.6f}   "
              f"chi=4 fid = {chi4_fids[0]:.6f}   "
              f"max_bond = {mps.max_bond()}")

    for L in range(num_layers):
        if verbose:
            print(f"\n---- chi=2 layer {L} ----")
        layer, layer_loss = optimise_one_layer_chi2(
            mps, loss, n_iter=n_iter, eps_init=eps_init, seed=L, verbose=verbose,
        )
        layers.append(layer)
        params.append(layer.get_params())

        new_mps = contract_layer_to_mps(layer, mps)
        for t in new_mps.tensors:
            t.modify(data=t.data.detach().clone())
        new_mps /= new_mps.norm()

        f2 = chi_fidelity(new_mps, 2)
        f4 = chi_fidelity(new_mps, 4)

        # assert f2 >= chi2_fids[-1] - mono_tol, (
        #     f"chi=2 fidelity DROPPED at layer {L}: "
        #     f"{chi2_fids[-1]:.8f} -> {f2:.8f}"
        # )
        # assert f4 >= chi4_fids[-1] - mono_tol, (
        #     f"chi=4 fidelity DROPPED at layer {L}: "
        #     f"{chi4_fids[-1]:.8f} -> {f4:.8f}"
        # )

        chi2_fids.append(f2)
        chi4_fids.append(f4)
        mps = new_mps
        mps_states.append(mps.copy())

        if verbose:
            print(f"[layer {L}] chi=2 fid = {f2:.6f}   chi=4 fid = {f4:.6f}   "
                  f"max_bond = {mps.max_bond()}   loss = {layer_loss:.4e}")

    return {
        "mps_states": mps_states,
        "params": params,
        "layers": layers,
        "chi2_fids": chi2_fids,
        "chi4_fids": chi4_fids,
    }

# --- Definition cell (from new.ipynb cell 58) ---
def su16_matrix(params: torch.Tensor) -> torch.Tensor:
    """255 real params -> SU(16) unitary, reshaped as a rank-8 tensor (2,)*8."""
    N = 16
    device = params.device
    dtype = torch.complex128

    Theta = torch.zeros((N, N), dtype=dtype, device=device)

    # Diagonal generators (Cartan subalgebra): N - 1 of them.
    for k in range(N - 1):
        vec = torch.zeros(N, dtype=torch.float64, device=device)
        vec[: k + 1] = 1.0
        vec[k + 1] = -(k + 1)
        Theta = Theta + params[k] * torch.diag(vec).to(dtype)

    # Off-diagonal generators (real-symmetric + imag-antisymmetric pairs):
    # N*(N-1) of them. Total params = (N-1) + N*(N-1) = N^2 - 1 = 255.
    idx = N - 1
    for i in range(N):
        for j in range(i + 1, N):
            mr = torch.zeros((N, N), dtype=dtype, device=device)
            mi = torch.zeros((N, N), dtype=dtype, device=device)
            mr[i, j] = mr[j, i] = 1.0
            mi[i, j] = 1.0j
            mi[j, i] = -1.0j
            Theta = Theta + params[idx] * mr + params[idx + 1] * mi
            idx += 2

    U = torch.linalg.matrix_exp(1j * Theta)
    return U.reshape(2, 2, 2, 2, 2, 2, 2, 2)


# Sanity: zero params -> exact identity, and the basis must give 255 params.
def _check_su16_identity():
    M = su16_matrix(torch.zeros(255, dtype=torch.float64)).reshape(16, 16)
    err = torch.linalg.norm(M - torch.eye(16, dtype=torch.complex128)).item()
    assert err < 1e-12, f"su16_matrix(0) != I (err={err:.3e})"
    # Also check perturbed version is unitary.
    p = 1e-2 * torch.randn(255, dtype=torch.float64)
    Mp = su16_matrix(p).reshape(16, 16)
    err2 = torch.linalg.norm(Mp.conj().T @ Mp - torch.eye(16, dtype=torch.complex128)).item()
    assert err2 < 1e-8, f"su16_matrix perturbed not unitary (err={err2:.3e})"
    print(f"OK: su16_matrix(0) == I  (err={err:.3e}); "
          f"perturbed unitarity err = {err2:.3e}")


_check_su16_identity()

# --- Definition cell (from new.ipynb cell 60) ---
def make_chi8_staircase(n: int, eps: float = 0.0,
                        seed: int | None = None) -> qtn.TensorNetwork:
    """
    Build the SU4 -> SU8 -> SU16 x (n-5) -> SU8 -> SU4 -> SU2 staircase.

    Each interior SU(16) carries three chi=2 wires (chi=8 in total) along
    the chain. The leading SU(4) -> SU(8) ramp grows the bus from 1 wire
    to 3 wires; the trailing SU(8) -> SU(4) -> SU(2) ramp brings it back
    down. With ``eps = 0`` every gate is the identity and the whole
    staircase is the identity map ``k{i} -> b{i}``.

    Parameters
    ----------
    n : int
        Number of qubits. Must be >= 6 so there is at least one middle
        SU(16) block.
    eps : float
        Std-dev of the Gaussian perturbation on the zero initialisation.
    seed : int | None
        RNG seed for the perturbation. ``None`` uses torch's default stream.
    """
    assert n >= 6, "Need n >= 6 for the SU4 -> SU8 -> SU16 ... -> SU8 -> SU4 -> SU2 staircase."

    if seed is not None:
        torch.manual_seed(seed)

    def _init(nparams):
        p = torch.zeros(nparams, dtype=torch.float64)
        if eps > 0:
            p = p + eps * torch.randn(nparams, dtype=torch.float64)
        p.requires_grad_(True)
        return p

    tensors = []
    k_free = 0
    b_free = 0
    c_counter = 0
    n_mid = n - 5  # number of interior SU(16) blocks

    # 1) Leading SU(4) on (k0, k1) -> (b0, c0).
    in_inds  = [f"k{k_free}", f"k{k_free + 1}"]; k_free += 2
    out_inds = [f"b{b_free}", f"c{c_counter}"]; b_free += 1; c_counter += 1
    tensors.append(PTensor(su4_matrix, _init(15),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU4", "L0"}))
    prev_c = [out_inds[1]]  # bus has 1 wire

    # 2) Leading SU(8) on (prev_c[0], k2, k3) -> (b1, c1, c2). Bus -> 2 wires.
    in_inds  = [prev_c[0], f"k{k_free}", f"k{k_free + 1}"]; k_free += 2
    out_inds = [f"b{b_free}",
                f"c{c_counter}", f"c{c_counter + 1}"]
    b_free += 1; c_counter += 2
    tensors.append(PTensor(su8_matrix, _init(63),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU8", "L1"}))
    prev_c = out_inds[1:]  # bus has 2 wires

    # 3) First SU(16) on (prev_c[0], prev_c[1], k4, k5) -> (b2, c3, c4, c5).
    #    Bus -> 3 wires.
    in_inds  = [prev_c[0], prev_c[1],
                f"k{k_free}", f"k{k_free + 1}"]; k_free += 2
    out_inds = [f"b{b_free}",
                f"c{c_counter}", f"c{c_counter + 1}", f"c{c_counter + 2}"]
    b_free += 1; c_counter += 3
    tensors.append(PTensor(su16_matrix, _init(255),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU16", "L2"}))
    prev_c = out_inds[1:]  # bus has 3 wires

    # 4) Remaining (n_mid - 1) middle SU(16)s.
    for i in range(1, n_mid):
        in_inds  = [prev_c[0], prev_c[1], prev_c[2], f"k{k_free}"]; k_free += 1
        out_inds = [f"b{b_free}",
                    f"c{c_counter}", f"c{c_counter + 1}", f"c{c_counter + 2}"]
        b_free += 1; c_counter += 3
        tensors.append(PTensor(su16_matrix, _init(255),
                               inds=tuple(in_inds + out_inds),
                               tags={"SU16", f"L{i + 2}"}))
        prev_c = out_inds[1:]

    # 5) Trailing SU(8) on (prev_c[0], prev_c[1], prev_c[2]) -> (b, c, c).
    #    Bus -> 2 wires.
    in_inds  = [prev_c[0], prev_c[1], prev_c[2]]
    out_inds = [f"b{b_free}",
                f"c{c_counter}", f"c{c_counter + 1}"]
    b_free += 1; c_counter += 2
    tensors.append(PTensor(su8_matrix, _init(63),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU8", f"L{n_mid + 2}"}))
    prev_c = out_inds[1:]  # bus has 2 wires

    # 6) Trailing SU(4) on (prev_c[0], prev_c[1]) -> (b, c). Bus -> 1 wire.
    in_inds  = [prev_c[0], prev_c[1]]
    out_inds = [f"b{b_free}", f"c{c_counter}"]; b_free += 1; c_counter += 1
    tensors.append(PTensor(su4_matrix, _init(15),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU4", f"L{n_mid + 3}"}))
    prev_c = [out_inds[1]]

    # 7) Trailing SU(2) on (prev_c[0]) -> b_{n-1}.
    in_inds  = [prev_c[0]]
    out_inds = [f"b{b_free}"]; b_free += 1
    tensors.append(PTensor(su2_matrix, _init(3),
                           inds=tuple(in_inds + out_inds),
                           tags={"SU2", f"L{n_mid + 4}"}))

    tn = qtn.TensorNetwork(tensors)
    return tn

# --- Definition cell (from new.ipynb cell 68) ---
def optimise_one_layer_chi8(mps: qtn.MatrixProductState,
                            loss_fn,
                            n_iter: int = 300,
                            tol: float = 1e-14,
                            eps_init: float = 1e-3,
                            seed: int | None = None,
                            verbose: bool = False):
    """
    Optimise a fresh chi=8 staircase to minimise ``loss_fn(layer, mps)``.

    Same contract as ``optimise_one_layer`` / ``optimise_one_layer_chi2``:
      - Returns the *best-loss* iterate seen during optimisation.
      - Falls back to the identity layer if the optimiser never beats it.
      - Returns ``(layer, best_loss)``.
    """
    n = mps.L
    layer = make_chi8_staircase(n, eps=eps_init, seed=seed)

    id_layer = make_chi8_staircase(n, eps=0.0)
    with torch.no_grad():
        id_loss   = float(loss_fn(id_layer, init_tn=mps))
        init_loss = float(loss_fn(layer,    init_tn=mps))
    if verbose:
        print(f"  identity loss = {id_loss:.6e}   "
              f"perturbed init loss = {init_loss:.6e}")

    best = {"val": float("inf"), "params": None}

    def _tracked(op, init_tn):
        val = loss_fn(op, init_tn=init_tn)
        scalar = float(val.detach()) if isinstance(val, torch.Tensor) else float(val)
        if scalar < best["val"]:
            params = op.get_params()
            best["val"] = scalar
            best["params"] = {k: _to_numpy(v).copy() for k, v in params.items()}
        return val

    tnopt = qtn.TNOptimizer(
        layer,
        _tracked,
        loss_constants={"init_tn": mps},
        optimizer="l-bfgs-b",
        autodiff_backend="torch",
        progbar=verbose,
    )
    _ = tnopt.optimize(n=n_iter, tol=tol)

    if best["params"] is not None:
        layer_best = layer.copy()
        layer_best.set_params({k: torch.as_tensor(v) for k, v in best["params"].items()})
        best_loss = best["val"]
    else:
        layer_best = layer
        with torch.no_grad():
            best_loss = float(loss_fn(layer_best, init_tn=mps))

    layer_best.apply_to_arrays(lambda x: torch.as_tensor(x))

    if best_loss > id_loss - 1e-12:
        if verbose:
            print(f"  [fallback] optimised loss {best_loss:.6e} >= identity "
                  f"loss {id_loss:.6e}; returning identity layer.")
        id_layer.apply_to_arrays(lambda x: torch.as_tensor(x))
        return id_layer, id_loss

    assert_layer_unitary_chi8(layer_best)
    return layer_best, best_loss

# --- Definition cell (from new.ipynb cell 70) ---
def sequential_disentangle_chi8(target_mps: qtn.MatrixProductState,
                                num_layers: int,
                                g: float = 0.0,
                                topk: int = 2,
                                n_iter: int = 300,
                                eps_init: float = 1e-3,
                                mono_tol: float = 1e-8,
                                verbose: bool = True) -> dict:
    """
    Run ``num_layers`` chi=8 staircases sequentially. Returns:

        {'mps_states', 'params', 'layers',
         'chi2_fids', 'chi4_fids', 'chi8_fids'}
    """
    mps = target_mps.copy()
    mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))

    loss = partial(tail_mass_loss, g=g, topk=topk)

    chi2_fids = [chi_fidelity(mps, 2)]
    chi4_fids = [chi_fidelity(mps, 4)]
    chi8_fids = [chi_fidelity(mps, 8)]
    layers = []
    params = []
    mps_states = [mps.copy()]

    if verbose:
        print(f"[init]   chi=2 fid = {chi2_fids[0]:.6f}   "
              f"chi=4 fid = {chi4_fids[0]:.6f}   "
              f"chi=8 fid = {chi8_fids[0]:.6f}   "
              f"max_bond = {mps.max_bond()}")

    for L in range(num_layers):
        if verbose:
            print(f"\n---- chi=8 layer {L} ----")
        layer, layer_loss = optimise_one_layer_chi8(
            mps, loss, n_iter=n_iter, eps_init=eps_init, seed=L, verbose=verbose,
        )
        layers.append(layer)
        params.append(layer.get_params())

        new_mps = contract_layer_to_mps(layer, mps)
        for t in new_mps.tensors:
            t.modify(data=t.data.detach().clone())
        new_mps /= new_mps.norm()

        f2 = chi_fidelity(new_mps, 2)
        f4 = chi_fidelity(new_mps, 4)
        f8 = chi_fidelity(new_mps, 8)

        # assert f2 >= chi2_fids[-1] - mono_tol, (
        #     f"chi=2 fidelity DROPPED at layer {L}: "
        #     f"{chi2_fids[-1]:.8f} -> {f2:.8f}"
        # )
        # assert f4 >= chi4_fids[-1] - mono_tol, (
        #     f"chi=4 fidelity DROPPED at layer {L}: "
        #     f"{chi4_fids[-1]:.8f} -> {f4:.8f}"
        # )
        # assert f8 >= chi8_fids[-1] - mono_tol, (
        #     f"chi=8 fidelity DROPPED at layer {L}: "
        #     f"{chi8_fids[-1]:.8f} -> {f8:.8f}"
        # )

        chi2_fids.append(f2)
        chi4_fids.append(f4)
        chi8_fids.append(f8)
        mps = new_mps
        mps_states.append(mps.copy())

        if verbose:
            print(f"[layer {L}] chi=2 fid = {f2:.6f}   "
                  f"chi=4 fid = {f4:.6f}   chi=8 fid = {f8:.6f}   "
                  f"max_bond = {mps.max_bond()}   loss = {layer_loss:.4e}")

    return {
        "mps_states": mps_states,
        "params": params,
        "layers": layers,
        "chi2_fids": chi2_fids,
        "chi4_fids": chi4_fids,
        "chi8_fids": chi8_fids,
    }

# --- Definition cell (from new.ipynb cell 152) ---
# === Warm-start Schmidt SSO helpers ===

import numpy as np
import torch
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor import PTensor
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import mps_to_circuit as mtc
import scipy.linalg as _scipy_linalg


# --- 1) Analytic Ran mapping (same path used in global_optimisation_claude.ipynb) ---
def _ws_qiskit_gate_info(qc_qiskit):
    gates = []
    for ci in qc_qiskit.data:
        instr = ci.operation
        qubits = [qc_qiskit.find_bit(q).index for q in ci.qubits]
        try:
            gates.append((instr.name, qubits, Operator(instr).data))
        except Exception:
            pass
    return gates


def _ws_ensure_mps_shape_lrp(mps):
    mps = mps.copy(); N = mps.L
    for i in range(N):
        T = mps[i]; p = mps.site_ind(i)
        virt = [ix for ix in T.inds if ix != p]
        if i == 0:    T.transpose(virt[0], p, inplace=True)
        elif i == N-1: T.transpose(virt[0], p, inplace=True)
        else:
            left = set(mps[i - 1].inds); right = set(mps[i + 1].inds)
            vL = next(ix for ix in virt if ix in left)
            vR = next(ix for ix in virt if ix in right)
            T.transpose(vL, vR, p, inplace=True)
    return mps


def mps_to_circuit_quimb_warm(mps):
    """Return (qc, gate_info) such that qc.psi == mps via the analytic
    mps-to-circuit decomposition. Same as global_optimisation_claude."""
    psi = _ws_ensure_mps_shape_lrp(mps.copy())
    psi.left_canonicalize(inplace=True)
    arrays = [np.asarray(A, dtype=np.complex128) for A in psi.arrays]
    qiskit_qc = mtc.mps_to_circuit(arrays, shape='lrp', method='exact')
    gates_info = _ws_qiskit_gate_info(qiskit_qc)
    qc = qtn.Circuit(mps.L)
    for name, qubits, U in gates_info:
        qc.apply_gate(U, *list(reversed(qubits)))
    return qc, gates_info


# --- 2) SU(N) parameterisations. ---
#
# IMPORTANT: SU(2), SU(4), SU(8), SU(16) all use the SAME generator basis
# (the `_ws_su_n_unitary` family). This is critical: an earlier version
# used Pauli (X, Y, Z) for SU(2) but the SU_N basis [Z, X, mi] for the
# generator-basis decomposition, causing recovered params to produce the
# WRONG matrix when fed through `_ws_su2`. The per-gate fit fid was 1.0
# (because both bases span SU(2)) but composed circuit fid dropped.
#
# UNIFIED CONVENTION (used everywhere):
#   Theta = sum_k params[k] * T_k,
#   T_0...T_{N-2} = traceless diagonal generators (diag([1,...,1,-(k+1),0,...]))
#   T_{N-1}...T_{N^2-2} = real off-diagonal (mr) and imag off-diagonal (mi) pairs.
#   U = exp(i Theta).

def _ws_su_n_unitary(params, N):
    device = params.device; dtype = torch.complex128
    Theta = torch.zeros((N, N), dtype=dtype, device=device)
    for k in range(N - 1):
        vec = torch.zeros(N, dtype=torch.float64, device=device)
        vec[: k + 1] = 1.0; vec[k + 1] = -(k + 1)
        Theta = Theta + params[k] * torch.diag(vec).to(dtype)
    idx = N - 1
    for i in range(N):
        for j in range(i + 1, N):
            mr = torch.zeros((N, N), dtype=dtype, device=device)
            mi = torch.zeros((N, N), dtype=dtype, device=device)
            mr[i, j] = 1.0; mr[j, i] = 1.0
            mi[i, j] = 1.0j; mi[j, i] = -1.0j
            Theta = Theta + params[idx] * mr + params[idx + 1] * mi
            idx += 2
    return torch.linalg.matrix_exp(1j * Theta)


# All SU(N) functions go through _ws_su_n_unitary so the basis is unified.
def _ws_su2(p):  return _ws_su_n_unitary(p, 2).reshape(2, 2)
def _ws_su4(p):  return _ws_su_n_unitary(p, 4).reshape(2,2,2,2)
def _ws_su8(p):  return _ws_su_n_unitary(p, 8).reshape(*([2]*6))
def _ws_su16(p): return _ws_su_n_unitary(p, 16).reshape(*([2]*8))


_WS_RANK = {
    2: ('SU2',  3,   ('k0', 'b0'), _ws_su2),
    4: ('SU4',  15,  ('k0', 'k1', 'b0', 'b1'), _ws_su4),
    6: ('SU8',  63,  ('k0', 'k1', 'k2', 'b0', 'b1', 'b2'), _ws_su8),
    8: ('SU16', 255, ('k0', 'k1', 'k2', 'k3', 'b0', 'b1', 'b2', 'b3'), _ws_su16),
}


def _ws_su_n_generators_np(N):
    """Return the N^2-1 Hermitian generators used inside ``_ws_su_n_unitary``.

    These are the SAME generators as the torch su_fn functions above, so the
    forward (params -> matrix) and inverse (matrix -> params) are CONSISTENT.
    """
    gens = []
    for k in range(N - 1):
        vec = np.zeros(N, dtype=np.float64)
        vec[: k + 1] = 1.0; vec[k + 1] = -(k + 1)
        gens.append(np.diag(vec).astype(np.complex128))
    for i in range(N):
        for j in range(i + 1, N):
            mr = np.zeros((N, N), dtype=np.complex128); mr[i, j] = 1.0; mr[j, i] = 1.0
            mi = np.zeros((N, N), dtype=np.complex128); mi[i, j] = 1.0j; mi[j, i] = -1.0j
            gens.append(mr); gens.append(mi)
    return gens


def _ws_params_from_unitary(U, N):
    """Recover params theta s.t. ``_ws_su_n_unitary(theta, N)`` equals U
    (modulo a global phase).

    Algorithm:
      1. Strip determinant: U_su = U / det(U)^{1/N}.
      2. Take matrix log: H = -i logm(U_su). (H may not be traceless if the
         principal-branch eigenvalues sum to a nonzero multiple of 2*pi.)
      3. FORCE H traceless (subtract Tr(H)/N * I). This costs a global phase
         on the reconstruction but lets us decompose H onto the SU(N)
         traceless-Hermitian generator basis.
      4. theta_k = Tr(T_k H) / Tr(T_k T_k) (generators are mutually
         orthogonal, so the Gram matrix is diagonal).

    The recon `_ws_su_n_unitary(theta, N)` matches U up to a global phase
    (an N-th root of unity). State fidelity is unaffected by this global
    phase; per-gate phases compose to a single global phase on the full
    circuit.
    """
    U = np.asarray(U).reshape(N, N).astype(np.complex128)
    det_U = np.linalg.det(U)
    # det_U^{1/N} via principal branch (np.power uses principal log).
    phase = det_U ** (1.0 / N)
    U_su = U / phase

    H = -1j * _scipy_linalg.logm(U_su)
    H = (H + H.conj().T) / 2  # symmetrise for numerical Hermiticity
    # Force traceless (handles the log-branch issue).
    H = H - (np.trace(H) / N) * np.eye(N, dtype=np.complex128)

    gens = _ws_su_n_generators_np(N)
    n_params = N * N - 1
    theta = np.zeros(n_params)
    for k in range(n_params):
        T = gens[k]
        gram_kk = float(np.real(np.trace(T @ T)))
        rhs_k   = float(np.real(np.trace(T @ H)))
        theta[k] = rhs_k / gram_kk
    return theta


# --- 3) Build the warm-start layer ---
#
# Two CRITICAL implementation choices (vs an earlier broken version):
#
# (a) We DO NOT canonicalise targ inds to canonical k0/b0/... labels and
#     then inverse-reindex. Instead, we read the target tensor's data
#     directly as a (N, N) matrix, recover params via _ws_params_from_unitary,
#     and build a PTensor with inds = targ.inds in the SAME axis order.
#     Why: quimb's qc.uni tensors have inds in some particular order with
#     internal wire labels; the old approach was sensitive to axis order
#     assumptions. The new approach is structurally identical to the
#     original gate (modulo a per-gate global phase from the log branch).
#
# (b) We store the **complex conjugate** ``conj(V_analytic)`` instead of
#     ``V_analytic`` itself. Why: new.ipynb's ``dense_state_from_op_mps``
#     applies the operator as ``V^T mps`` (matrix transpose, not adjoint),
#     because it just relabels k<->b and contracts. For REAL V this equals
#     ``V^{-1} mps`` (V is orthogonal), so applying it to ``V|0>`` gives
#     ``|0>``. For COMPLEX V, ``V^T V != I`` and the action is wrong.
#     By storing ``conj(V)``, the dense_state_from_op_mps action becomes
#     ``(conj V)^T mps = V^dag mps``, which IS the proper adjoint, so
#     applied to ``V|0>`` it gives ``|0>``. Verified on bond-8 random:
#     layer 1's approx is real (the original code worked), but layer 2+'s
#     approx becomes complex (chi-trunc of dense-rebuilt residual), where
#     storing V (the old way) gave only 0.85 overlap with |0> vs 1.000 with
#     conj(V).

def build_warm_layer(current_mps, chi, verbose=False):
    """Build the warm-init parameterised disentangler for chi-bond SSO.

    The warm-start layer is the parameterised representation of the
    analytic Ran circuit V where V|0> = chi-truncation of current_mps:

      approx        = compress(current_mps, max_bond=chi)
      V             = mps_to_circuit_quimb(approx)        # V|0> = approx (EXACT)
      warm_layer    = (PTensor replacement of V's gates)  # V_param ~ V (mod global phase)

    By construction (when applied via new.ipynb's `dense_state_from_op_mps`,
    which contracts the op acting AS V^dag — its effect on state):

      warm_layer  applied to  approx        -> |0...0>   (EXACT)
      warm_layer  applied to  current_mps   -> a state with
                                                |<.|0...0>|^2 = chi-truncation fidelity

    With ``verbose=True`` every step prints a diagnostic so you can
    interrogate whether each piece is working as intended. In particular:
      STEP 1: chi-truncation fidelity
      STEP 2: analytic V|0> exactly equals approx
      STEP 3: composed parameterised V_param|0> matches V_analytic|0>
      STEP 3a: V_param|0> matches approx
      STEP 3b: warm_layer acts as the chi-truncation disentangler
    """
    import numpy as _np
    n = current_mps.L

    # ============================================================
    # STEP 1: Truncate current_mps to bond=chi  ->  approx.
    # ============================================================
    approx = current_mps.copy()
    approx.apply_to_arrays(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else _np.asarray(x))
    approx.compress(max_bond=chi)
    approx /= approx.norm()

    if verbose:
        ref = current_mps.copy()
        ref.apply_to_arrays(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else _np.asarray(x))
        ref_d = ref.to_dense().reshape(-1); ref_d /= _np.linalg.norm(ref_d)
        approx_d = approx.to_dense().reshape(-1); approx_d /= _np.linalg.norm(approx_d)
        trunc_fid = float(abs(_np.vdot(approx_d, ref_d)) ** 2)
        print(f"    [warm STEP 1] chi={chi} truncation:")
        print(f"      current_mps max_bond = {ref.max_bond()}    approx max_bond = {approx.max_bond()}")
        print(f"      |<approx | current_mps>|^2 = {trunc_fid:.10f}  (truncation fidelity)")

    # ============================================================
    # STEP 2: Build analytic forward circuit V s.t. V|0> = approx.
    # ============================================================
    qc, gate_info = mps_to_circuit_quimb_warm(approx)
    qc_uni = qc.uni

    if verbose:
        psi_built = qc.psi
        psi_built_d = psi_built.to_dense().reshape(-1); psi_built_d /= _np.linalg.norm(psi_built_d)
        fid_VzeroToApprox = float(abs(_np.vdot(psi_built_d, approx_d)) ** 2)
        print(f"    [warm STEP 2] analytic forward circuit V (mps_to_circuit_quimb):")
        print(f"      {len(gate_info)} gates total")
        print(f"      |<V|0> | approx>|^2 = {fid_VzeroToApprox:.12f}  "
              f"(EXPECT ~1.0 to machine precision)")
        if fid_VzeroToApprox < 0.999:
            print(f"      !!! WARNING: analytic V|0> != approx; fid = {fid_VzeroToApprox}")

    # ============================================================
    # STEP 3: For each tensor in qc.uni, build a PTensor with EXACT params
    # (via _ws_params_from_unitary) and SAME inds. This is structurally
    # identical to the original gate (modulo a per-gate global phase).
    # ============================================================
    new_tensors = []
    per_gate_fits = []
    for k, targ in enumerate(qc_uni.tensors):
        rank = len(targ.inds)
        tag, n_params, canon, su_fn = _WS_RANK[rank]
        N_dim = 2 ** (rank // 2)

        # Read targ.data as a (N_dim, N_dim) matrix in axis order of targ.inds,
        # then take complex conjugate (see header comment (b)). The stored
        # warm_layer represents conj(V_analytic); dense_state_from_op_mps's
        # V^T-style action turns this into V_analytic^dag (the proper
        # disentangler).
        U_targ = _np.asarray(targ.data).reshape(N_dim, N_dim).astype(_np.complex128)
        U_targ_conj = U_targ.conj()
        theta = _ws_params_from_unitary(U_targ_conj, N_dim)
        params_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)

        # Per-gate state fidelity (vs the conjugated target -- this is what
        # the PTensor should match).
        recon = su_fn(params_t).detach().cpu().numpy().reshape(N_dim, N_dim)
        hs = _np.vdot(recon.ravel(), U_targ_conj.ravel())
        per_gate_fid = abs(hs) ** 2 / (_np.linalg.norm(recon) ** 2 * _np.linalg.norm(U_targ_conj) ** 2)
        per_gate_fits.append(per_gate_fid)

        # Build PTensor with EXACT same inds (no canonical reindex).
        pt = PTensor(su_fn, params_t, inds=targ.inds, tags=[tag])
        new_tensors.append(pt)

    warm_layer = qtn.TensorNetwork(new_tensors)

    if verbose:
        worst_gate = min(per_gate_fits)
        avg_gate = sum(per_gate_fits) / len(per_gate_fits)
        print(f"    [warm STEP 3] PTensor replacement (exact params):")
        print(f"      per-gate state fid: worst = {worst_gate:.12f}  "
              f"avg = {avg_gate:.12f}  ({len(per_gate_fits)} gates)")

        # STEP 3a: composed parameterised V_param vs analytic V_analytic.
        # warm_layer stores conj(V_analytic). To get V_param|0>, we need to
        # contract conj(warm_layer) with |0>. Use .H (which conjugates data
        # without swapping inds) and then contract.
        psi_analytic = qc.psi
        psi_analytic_d = psi_analytic.to_dense().reshape(-1)
        psi_analytic_d /= _np.linalg.norm(psi_analytic_d)
        zero_mps = qtn.MPS_computational_state('0' * n)
        zero_mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
        # warm_layer stores conj(V). warm_layer.H conjugates data -> V.
        # Then (V | zero_at_k).contract(output=b) gives V|0>_at_b.
        Vparam_zero = (warm_layer.H | zero_mps).contract(
            backend='torch', output_inds=tuple(f'b{j}' for j in range(n)),
        )
        vp_d = Vparam_zero.data.detach().cpu().numpy().reshape(-1)
        vp_d /= _np.linalg.norm(vp_d)
        fid_paramVS_analytic = float(abs(_np.vdot(vp_d, psi_analytic_d)) ** 2)
        fid_paramVS_approx = float(abs(_np.vdot(vp_d, approx_d)) ** 2)
        print(f"    [warm STEP 3a] composed parameterised vs analytic:")
        print(f"      |<V_param|0> | V_analytic|0>|^2 = {fid_paramVS_analytic:.12f}  "
              f"(EXPECT ~1.0)")
        print(f"      |<V_param|0> | approx>|^2       = {fid_paramVS_approx:.12f}  "
              f"(EXPECT ~1.0)")
        if fid_paramVS_analytic < 0.999:
            print(f"      !!! WARNING: composed circuit drift > 1e-3; "
                  f"fid = {fid_paramVS_analytic}")

        # STEP 3b: action of warm_layer as the disentangler via
        # dense_state_from_op_mps. Should give |0...0> exactly when fed
        # approx (the chi-truncation that V_analytic prepares from |0>).
        # For complex circuits, this only works with the conj(V) storage
        # convention (see header comment (b)).
        approx_t = approx.copy()
        approx_t.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
        s = dense_state_from_op_mps(warm_layer, approx_t).detach().cpu().numpy().reshape(-1)
        s /= _np.linalg.norm(s)
        overlap_approx = float(abs(s[0]) ** 2)

        current_t = current_mps.copy()
        if not isinstance(current_t[0].data, torch.Tensor):
            current_t.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
        sc = dense_state_from_op_mps(warm_layer, current_t).detach().cpu().numpy().reshape(-1)
        sc /= _np.linalg.norm(sc)
        overlap_current = float(abs(sc[0]) ** 2)
        print(f"    [warm STEP 3b] disentangler action via dense_state_from_op_mps:")
        print(f"      warm_layer | approx      -> |<.|0...0>|^2 = "
              f"{overlap_approx:.12f}  (EXPECT ~1.0)")
        print(f"      warm_layer | current_mps -> |<.|0...0>|^2 = "
              f"{overlap_current:.12f}  (EXPECT ~truncation fid {trunc_fid:.6f})")

    return warm_layer

# --- Definition cell (from new.ipynb cell 156) ---
# === Warm-start sequential disentangler ===
from functools import partial


def optimise_warm_layer(mps, chi, n_iter=300, verbose=False):
    """Build warm layer from `mps` at bond=chi, then optimise it against
    tail_mass_loss(topk=chi). Returns (opt_layer, init_loss, final_loss).

    Safety: among {opt_layer, warm_layer, id_layer} pick whichever has
    the LOWEST tail_mass loss. This guarantees final_loss <= min(warm,
    identity) at every layer.
    """
    warm_layer = build_warm_layer(mps, chi, verbose=verbose)

    mps_t = mps.copy()
    mps_t.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    loss_fn = partial(tail_mass_loss, g=0.0, topk=chi)

    init_loss = float(loss_fn(warm_layer, init_tn=mps_t))

    # Identity baseline (the upper bound on opt's output loss).
    if chi == 2:
        id_layer = make_chi2_staircase(mps.L, eps=0.0)
    elif chi == 4:
        id_layer = make_hybrid_staircase(mps.L, eps=0.0)
    else:
        id_layer = make_chi8_staircase(mps.L, eps=0.0)
    id_loss = float(loss_fn(id_layer, init_tn=mps_t))

    if verbose:
        print(f"    [warm-opt] init loss (warm) = {init_loss:.6e}   "
              f"id loss = {id_loss:.6e}")

    tnopt = qtn.TNOptimizer(
        warm_layer, loss_fn,
        loss_constants={'init_tn': mps_t},
        optimizer='l-bfgs-b', autodiff_backend='torch', progbar=False,
    )
    opt_layer_lbfgs = tnopt.optimize(n=n_iter, tol=1e-14)
    final_loss_lbfgs = float(loss_fn(opt_layer_lbfgs, init_tn=mps_t))

    # Three-way safety: pick whichever of {opt, warm, id} has the lowest loss.
    candidates = [
        ('opt',  opt_layer_lbfgs, final_loss_lbfgs),
        ('warm', warm_layer,      init_loss),
        ('id',   id_layer,        id_loss),
    ]
    best_name, best_layer, best_loss = min(candidates, key=lambda c: c[2])
    if verbose:
        if best_name != 'opt':
            print(f"    [warm-opt] SAFETY: L-BFGS-B final ({final_loss_lbfgs:.4e}) "
                  f"is not the lowest; using {best_name} (loss={best_loss:.4e}) instead.")
        else:
            print(f"    [warm-opt] final loss = {best_loss:.6e}  "
                  f"(gain over warm init = {init_loss - best_loss:+.4e})")
    return best_layer, init_loss, best_loss


def sequential_disentangle_warm(target_mps, chi, num_layers,
                                 n_iter=300, verbose=False):
    """K-iteration warm-start sequential disentangler at fixed chi.

    Returns {'layers', 'chi_fids' (per layer), 'init_losses', 'final_losses'}.
    """
    mps = target_mps.copy()
    mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))

    chi_fids = [chi_fidelity(mps, chi)]
    layers = []
    init_losses = []
    final_losses = []

    if verbose:
        print(f"  [init] chi={chi} fid = {chi_fids[0]:.6f}   "
              f"max_bond = {mps.max_bond()}")

    for L in range(num_layers):
        if verbose:
            print(f"\n  ---- layer {L+1}/{num_layers} (warm-start) ----")
        # Make a numpy-side copy of mps for build_warm_layer (which
        # calls compress + mps_to_circuit_quimb on numpy arrays).
        mps_np = mps.copy()
        mps_np.apply_to_arrays(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x))

        # chi_fid of the input mps -- baseline to monotonicity-check the layer.
        input_chi_fid = chi_fidelity(mps_np, chi)

        layer, init_loss, final_loss = optimise_warm_layer(
            mps_np, chi, n_iter=n_iter, verbose=verbose,
        )

        # Apply opt layer to current mps -> new mps.
        new_state = dense_state_from_op_mps(layer, mps)
        new_state = new_state.detach()
        new_state_np = new_state.cpu().numpy().reshape(-1)
        new_state_np = new_state_np / np.linalg.norm(new_state_np)
        new_mps = qtn.MatrixProductState.from_dense(
            new_state_np, dims=[2] * target_mps.L,
        )
        new_mps /= new_mps.norm()

        # === Monotonicity safety net (per-layer chi_fid check) ===
        # If applying the optimised layer would DROP chi_fid (the metric
        # we ultimately care about), revert to the identity layer for
        # this iteration. Identity-applied gives back the input mps
        # unchanged, so chi_fid is preserved.
        output_chi_fid_pre = chi_fidelity(new_mps, chi)
        if output_chi_fid_pre < input_chi_fid - 1e-6:
            if verbose:
                print(f"    [post-apply SAFETY] chi_fid would drop: "
                      f"{input_chi_fid:.6f} -> {output_chi_fid_pre:.6f}; "
                      f"reverting THIS LAYER to identity (mps unchanged).")
            # Identity layer applied to mps gives mps back; just keep mps.
            new_mps = mps_np.copy()
            new_mps /= new_mps.norm()
            # Replace the stored "layer" with the identity one so the
            # final rebuild circuit reflects what was actually applied.
            if chi == 2:
                layer = make_chi2_staircase(target_mps.L, eps=0.0)
            elif chi == 4:
                layer = make_hybrid_staircase(target_mps.L, eps=0.0)
            else:
                layer = make_chi8_staircase(target_mps.L, eps=0.0)

        layers.append(layer)
        init_losses.append(init_loss)
        final_losses.append(final_loss)

        # Update mps (as torch).
        mps = new_mps.copy()
        mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))

        f_chi = chi_fidelity(new_mps, chi)
        chi_fids.append(f_chi)
        if verbose:
            print(f"    [post layer {L+1}] chi={chi} fid = {f_chi:.6f}   "
                  f"max_bond = {new_mps.max_bond()}   "
                  f"(input was {input_chi_fid:.6f}, "
                  f"{'KEPT layer' if f_chi >= input_chi_fid - 1e-6 else 'identity-reverted'})")

    return {
        'chi':           chi,
        'num_layers':    num_layers,
        'layers':        layers,
        'chi_fids':      chi_fids,
        'init_losses':   init_losses,
        'final_losses':  final_losses,
        'final_mps':     mps,
    }


print('Warm-start helpers loaded.')
print('  build_warm_layer / optimise_warm_layer / sequential_disentangle_warm.')

# --- Definition cell (from new.ipynb cell 224) ---
def image_to_mps(img_array, max_bond=None, cutoff=0.0):
    """Convert a 2D image (H, W) to an MPS.

    Flatten row-by-row (C-order), pad with zeros to next power of 2,
    normalise to a state vector, then build the MPS. Optionally
    compress to ``max_bond``.

    IMPORTANT: ``from_dense`` defaults to ``cutoff=1e-10`` which silently
    drops tiny Schmidt values during the SVD chain. For large images
    (e.g. 128x128 has 13 cuts) the accumulated error can reach ~1e-6.
    We pass ``cutoff=0.0`` to keep every Schmidt component for an exact
    full-bond representation. Compression then happens only if the
    caller asks for it via ``max_bond``.

    Returns ``(mps, meta)``. ``meta`` records everything needed for the
    inverse: original ``shape``, the pre-normalisation ``norm``, number
    of qubits ``n_qubits``, and ``padded_len``.
    """
    img = np.asarray(img_array, dtype=np.float64)
    H, W = img.shape
    vec = img.flatten()                         # row-by-row (C-order)
    L = len(vec)
    n = int(np.ceil(np.log2(L)))
    padded_len = 2 ** n

    state = np.zeros(padded_len, dtype=np.complex128)
    state[:L] = vec
    norm = float(np.linalg.norm(state))
    if norm > 0:
        state = state / norm

    # cutoff=0.0 -> keep ALL Schmidt components (no implicit compression).
    mps = qtn.MatrixProductState.from_dense(state, dims=[2] * n, cutoff=0.0)
    if max_bond is not None:
        mps.compress(max_bond=max_bond, cutoff=cutoff)

    meta = {
        'shape':       (H, W),
        'norm':        norm,
        'n_qubits':    n,
        'padded_len':  padded_len,
    }
    return mps, meta


def mps_to_image(mps, meta):
    """Inverse of ``image_to_mps``. Returns a (H, W) numpy array.

    If the MPS was compressed (so its state-vector norm has drifted
    from 1), we renormalise back to the tracked ``meta['norm']`` so
    the recovered image scale matches the original.
    """
    H, W = meta['shape']
    state = np.asarray(mps.to_dense()).reshape(-1)
    # Compressed MPS may not be unit-norm; re-scale to tracked norm.
    cur_norm = float(np.linalg.norm(state))
    if cur_norm > 0:
        state = state * (meta['norm'] / cur_norm)
    else:
        state = state * meta['norm']
    vec = state[:H * W].real
    return vec.reshape(H, W)


print("Image <-> MPS helpers defined: image_to_mps, mps_to_image.")

# --- Definition cell (from new.ipynb cell 234) ---
# Gate-by-gate state propagation helpers, plus the SSO replay /
# reconstruction utilities. Use the warm-start "stored = conj(V)"
# convention so the dispatch matches dense_state_from_op_mps exactly.
#
# Smoke-verified (cf. _smoke_image_sso.py):
#   1. gate-by-gate V^dag MATCHES dense_state_from_op_mps to 1e-15.
#   2. SSO reconstruction fid vs target MATCHES recorded chi_fid to 1e-15.


def _img_apply_gate_to_psi(psi, U, qubits, n_qubits):
    """Apply a 2^q x 2^q gate U at ``qubits`` on the state psi (shape (2,)*n)."""
    q = len(qubits)
    N = 2 ** q
    perm = list(qubits) + [i for i in range(n_qubits) if i not in qubits]
    psi_p = np.transpose(psi, perm).reshape(N, -1)
    psi_p = U @ psi_p
    psi_p = psi_p.reshape([2] * q + [2] * (n_qubits - q))
    inv = [0] * n_qubits
    for i, p in enumerate(perm):
        inv[p] = i
    return np.transpose(psi_p, inv)


def _img_apply_layer_V_dag(state_flat, layer, gate_info, n_qubits):
    """Apply a layer's V^dag (disentangler) gate-by-gate to a state.

    Stored data per gate = ``conj(V_gate)``; V_gate^dag = stored.T.
    Within a layer, V^dag applies gates in REVERSE within-layer order.
    """
    psi = state_flat.reshape([2] * n_qubits)
    for i in range(len(gate_info) - 1, -1, -1):
        t = layer.tensors[i]
        rank = len(t.inds); N = 2 ** (rank // 2)
        arr = t.data.detach().cpu().numpy() if hasattr(t.data, 'detach') else np.asarray(t.data)
        U_apply = arr.reshape(N, N).T            # V^dag
        _, qubits, _ = gate_info[i]
        psi = _img_apply_gate_to_psi(psi, U_apply, list(reversed(qubits)), n_qubits)
    return psi.flatten()


def _img_apply_layer_V(state_flat, layer, gate_info, n_qubits):
    """Apply a layer's V (forward) gate-by-gate. V_gate = stored.conj().
    Within a layer, V applies gates in FORWARD within-layer order."""
    psi = state_flat.reshape([2] * n_qubits)
    for i, (_, qubits, _) in enumerate(gate_info):
        t = layer.tensors[i]
        rank = len(t.inds); N = 2 ** (rank // 2)
        arr = t.data.detach().cpu().numpy() if hasattr(t.data, 'detach') else np.asarray(t.data)
        U_apply = arr.reshape(N, N).conj()       # V
        psi = _img_apply_gate_to_psi(psi, U_apply, list(reversed(qubits)), n_qubits)
    return psi.flatten()


def img_sso_replay(target_mps, layers, chi):
    """Replay the warm-start SSO sweep gate-by-gate.

    Returns (gate_infos, residual_states, bond_dims, chi_fids):
      * gate_infos[L] = analytic gate_info used at layer L (qubit assignments).
      * residual_states[L] = dense state vector AFTER applying layer L.
      * bond_dims[L] = max_bond of the MPS form of residual_states[L]
        (using cutoff=0.0 so this reflects the true rank, not a cutoff
        artefact).
      * chi_fids[L] = global chi-trunc fid of residual_states[L].
    residual_states has length K+1 (index 0 = the input target).
    """
    n = target_mps.L
    state = target_mps.to_dense().reshape(-1).astype(np.complex128)
    state = state / np.linalg.norm(state)

    residual_states = [state.copy()]
    bond_dims = []
    chi_fids = []
    gate_infos = []

    # Initial bond_dim / chi_fid (before any disentangler).
    mps0 = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=0.0); mps0 /= mps0.norm()
    bond_dims.append(mps0.max_bond())
    chi_fids.append(float(chi_fidelity(mps0, chi)))

    for L, layer in enumerate(layers):
        # Build MPS to get analytic gate_info at this layer.
        mps_now = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=0.0)
        mps_now /= mps_now.norm()
        approx = mps_now.copy(); approx.compress(max_bond=chi, cutoff=0.0)
        approx /= approx.norm()
        _, gi = mps_to_circuit_quimb_warm(approx)
        gate_infos.append(gi)

        # Apply V^dag gate-by-gate to advance the state.
        state = _img_apply_layer_V_dag(state, layer, gi, n)
        state = state / np.linalg.norm(state)
        residual_states.append(state.copy())

        # Bond + chi_fid of new residual.
        mps_step = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=0.0)
        mps_step /= mps_step.norm()
        bond_dims.append(mps_step.max_bond())
        chi_fids.append(float(chi_fidelity(mps_step, chi)))

    return {
        'gate_infos':       gate_infos,
        'residual_states':  residual_states,
        'bond_dims':        bond_dims,
        'chi_fids':         chi_fids,
    }


def img_sso_reconstruct(target_mps, layers, gate_infos, chi, K):
    """Compute the SSO forward-reconstruction state vector at (chi, K).

    1. Forward-disentangle target through first K layers.
    2. chi-truncate the residual.
    3. Apply each of those K layers in REVERSE order, each as forward V.

    The resulting state has fidelity ``chi_fid`` vs the target (= the
    recorded chi_fid at layer K).
    """
    n = target_mps.L
    layers_used = layers[:K]
    gate_infos_used = gate_infos[:K]

    # Step 1: forward V^dag through K layers.
    state = target_mps.to_dense().reshape(-1).astype(np.complex128)
    state = state / np.linalg.norm(state)
    for L, (layer, gi) in enumerate(zip(layers_used, gate_infos_used)):
        state = _img_apply_layer_V_dag(state, layer, gi, n)
        state = state / np.linalg.norm(state)

    # Step 2: chi-truncate the final residual.
    final_mps = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=0.0)
    final_mps /= final_mps.norm()
    final_mps.compress(max_bond=chi, cutoff=0.0)
    final_mps /= final_mps.norm()
    state = final_mps.to_dense().reshape(-1)
    state = state / np.linalg.norm(state)

    # Step 3: reverse-apply layers, each as forward V.
    for L_idx in range(K - 1, -1, -1):
        layer = layers_used[L_idx]
        gi = gate_infos_used[L_idx]
        state = _img_apply_layer_V(state, layer, gi, n)
        state = state / np.linalg.norm(state)

    return state


print("Image-SSO helpers defined: img_sso_replay, img_sso_reconstruct, "
      "_img_apply_gate_to_psi, _img_apply_layer_V, _img_apply_layer_V_dag.")

# --- Definition cell (from new.ipynb cell 240) ---
from qiskit import QuantumCircuit, transpile
import time


def build_sso_reconstruction_qiskit_circuit(target_mps, layers, gate_infos, chi):
    """Build a Qiskit QuantumCircuit that implements the full SSO
    reconstruction at (chi, K=len(layers))."""
    n = target_mps.L

    # Step 1: forward V^dag through all layers to get final residual.
    state = target_mps.to_dense().reshape(-1).astype(np.complex128)
    state = state / np.linalg.norm(state)
    for L, (layer, gi) in enumerate(zip(layers, gate_infos)):
        state = _img_apply_layer_V_dag(state, layer, gi, n)
        state = state / np.linalg.norm(state)

    # Step 2: chi-truncate the final residual and build its analytic prep circuit.
    final_mps = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=0.0)
    final_mps /= final_mps.norm()
    final_chi_trunc = final_mps.copy()
    final_chi_trunc.compress(max_bond=chi, cutoff=0.0)
    final_chi_trunc /= final_chi_trunc.norm()
    _, prep_gate_info = mps_to_circuit_quimb_warm(final_chi_trunc)

    # Step 3: assemble Qiskit circuit.
    qc = QuantumCircuit(n)

    # Apply prep gates (analytic V_prep |0> = final_chi_trunc).
    for _, qubits, U in prep_gate_info:
        qc.unitary(U, list(reversed(qubits)))

    # Apply layers in REVERSE order, each as forward V (gates in forward
    # within-layer order with U_apply = stored.conj()).
    for L_idx in range(len(layers) - 1, -1, -1):
        layer = layers[L_idx]
        gi = gate_infos[L_idx]
        for i, (_, qubits, _) in enumerate(gi):
            t = layer.tensors[i]
            rank = len(t.inds); N = 2 ** (rank // 2)
            arr = t.data.detach().cpu().numpy() if hasattr(t.data, 'detach') else np.asarray(t.data)
            U_apply = arr.reshape(N, N).conj()
            qc.unitary(U_apply, list(reversed(qubits)))

    return qc

# --- Definition cell (from new.ipynb cell 284) ---
# Modified replay that uses a configurable cutoff (default 1e-12)
# in the MPS reconstruction step. With cutoff=1e-12 the bond drops to
# the EFFECTIVE rank (number of Schmidt values above 1e-12 * largest).

def img_sso_replay_with_cutoff(target_mps, layers, chi, cutoff=1e-12):
    """Same as img_sso_replay but uses a non-zero cutoff in
    ``MatrixProductState.from_dense`` calls. Returns the same dict
    with 'bond_dims' reflecting the effective bond, not the full rank."""
    n = target_mps.L
    state = target_mps.to_dense().reshape(-1).astype(np.complex128)
    state = state / np.linalg.norm(state)

    residual_states = [state.copy()]
    bond_dims = []
    chi_fids = []
    gate_infos = []

    mps0 = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=cutoff)
    mps0 /= mps0.norm()
    bond_dims.append(mps0.max_bond())
    chi_fids.append(float(chi_fidelity(mps0, chi)))

    for L, layer in enumerate(layers):
        # gate_info from analytic chi-trunc (always cutoff=0 here for exactness).
        mps_now = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=0.0)
        mps_now /= mps_now.norm()
        approx = mps_now.copy(); approx.compress(max_bond=chi, cutoff=0.0)
        approx /= approx.norm()
        _, gi = mps_to_circuit_quimb_warm(approx)
        gate_infos.append(gi)

        state = _img_apply_layer_V_dag(state, layer, gi, n)
        state = state / np.linalg.norm(state)
        residual_states.append(state.copy())

        # Effective bond uses the configurable cutoff.
        mps_step = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=cutoff)
        mps_step /= mps_step.norm()
        bond_dims.append(mps_step.max_bond())
        chi_fids.append(float(chi_fidelity(mps_step, chi)))

    return {
        'gate_infos':       gate_infos,
        'residual_states':  residual_states,
        'bond_dims':        bond_dims,
        'chi_fids':         chi_fids,
        'cutoff':           cutoff,
    }


def reconstruction_bond_growth(target_mps, layers, gate_infos, chi, cutoff=1e-12):
    """Apply inverse layers to chi-trunc, recording bond after each
    inverse layer is applied. Returns a list of length K+1 with the
    bond starting at chi and growing as inverse layers are applied."""
    n = target_mps.L

    # Forward to get final residual, then chi-trunc.
    state = target_mps.to_dense().reshape(-1).astype(np.complex128)
    state = state / np.linalg.norm(state)
    for layer, gi in zip(layers, gate_infos):
        state = _img_apply_layer_V_dag(state, layer, gi, n)
        state = state / np.linalg.norm(state)
    final_mps = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=0.0)
    final_mps /= final_mps.norm()
    final_mps.compress(max_bond=chi, cutoff=0.0)
    final_mps /= final_mps.norm()
    state = final_mps.to_dense().reshape(-1)
    state = state / np.linalg.norm(state)

    bonds = [chi]   # starting bond is chi (chi-trunc)

    # Apply each inverse layer in reverse order; record bond after each.
    for L_idx in range(len(layers) - 1, -1, -1):
        state = _img_apply_layer_V(state, layers[L_idx], gate_infos[L_idx], n)
        state = state / np.linalg.norm(state)
        mps_step = qtn.MatrixProductState.from_dense(state, dims=[2]*n, cutoff=cutoff)
        mps_step /= mps_step.norm()
        bonds.append(mps_step.max_bond())
    return bonds


print("Heis16-SSO bond helpers defined: img_sso_replay_with_cutoff, "
      "reconstruction_bond_growth.")

# Identity init = warm STRUCTURE + eps-perturbed identity params.

def build_identity_layer_with_warm_structure(target_mps, chi, eps=1e-3,
                                                seed=None, verbose=False):
    """Identical gate structure to ``build_warm_layer(target_mps, chi)``
    but every PTensor's params reset to ``eps * randn`` (so the layer
    is near-identity instead of near the analytic warm V^dag)."""
    if seed is not None:
        torch.manual_seed(seed)
    init_layer = build_warm_layer(target_mps, chi, verbose=False)
    for t in init_layer.tensors:
        # PTensor stores .params (a torch tensor of free parameters).
        if hasattr(t, 'params'):
            t.params = eps * torch.randn_like(t.params)
    if verbose:
        print(f"    [id-init] built identity-like layer with warm structure, "
              f"eps={eps}, seed={seed}")
    return init_layer


def optimise_identity_layer(mps, chi, eps=1e-3, seed=None, n_iter=300, verbose=False):
    """Identity-init version of ``optimise_warm_layer``: starts L-BFGS-B
    from the eps-perturbed identity layer and runs the same loss."""
    init_layer = build_identity_layer_with_warm_structure(
        mps, chi, eps=eps, seed=seed, verbose=verbose,
    )
    mps_t = mps.copy()
    mps_t.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))

    from functools import partial as _partial
    loss_fn = _partial(tail_mass_loss, g=0.0, topk=chi)
    init_loss = float(loss_fn(init_layer, init_tn=mps_t))

    tnopt = qtn.TNOptimizer(
        init_layer, loss_fn,
        loss_constants={'init_tn': mps_t},
        optimizer='l-bfgs-b', autodiff_backend='torch', progbar=False,
    )
    opt_layer = tnopt.optimize(n=n_iter, tol=1e-14)
    final_loss = float(loss_fn(opt_layer, init_tn=mps_t))
    if verbose:
        print(f"    [id-opt] init = {init_loss:.4e}   final = {final_loss:.4e}   "
              f"gain = {init_loss - final_loss:+.4e}")
    # 2-way safety: pick best of (opt, init).
    if final_loss <= init_loss:
        return opt_layer, init_loss, final_loss
    else:
        if verbose:
            print(f"    [id-opt] SAFETY: opt is worse than init; using init.")
        return init_layer, init_loss, init_loss


def sequential_disentangle_identity_struct(target_mps, chi, num_layers,
                                              n_iter=300, eps=1e-3, seed=None,
                                              verbose=False):
    """Identity-init version of ``sequential_disentangle_warm``: each
    layer is initialised as eps-perturbed identity with warm gate
    structure, then optimised against tail_mass."""
    mps = target_mps.copy()
    mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    chi_fids = [chi_fidelity(mps, chi)]
    layers = []
    init_losses = []
    final_losses = []
    if verbose:
        print(f"  [id-init] chi={chi} fid = {chi_fids[0]:.6f}   "
              f"max_bond = {mps.max_bond()}")
    for L in range(num_layers):
        if verbose:
            print(f"\n  ---- layer {L+1}/{num_layers} (identity-init) ----")
        mps_np = mps.copy()
        mps_np.apply_to_arrays(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x))
        input_chi_fid = chi_fidelity(mps_np, chi)

        layer, init_loss, final_loss = optimise_identity_layer(
            mps_np, chi, eps=eps, seed=seed if seed is None else seed + L,
            n_iter=n_iter, verbose=verbose,
        )

        # Apply layer to advance mps.
        new_state = dense_state_from_op_mps(layer, mps).detach().cpu().numpy().reshape(-1)
        new_state = new_state / np.linalg.norm(new_state)
        new_mps = qtn.MatrixProductState.from_dense(new_state, dims=[2] * target_mps.L)
        new_mps /= new_mps.norm()
        # Monotonicity safety: revert if chi_fid would drop.
        output_chi_fid = chi_fidelity(new_mps, chi)
        if output_chi_fid < input_chi_fid - 1e-6:
            if verbose:
                print(f"    [id-init SAFETY] chi_fid would drop "
                      f"{input_chi_fid:.6f} -> {output_chi_fid:.6f}; reverting.")
            new_mps = mps_np.copy(); new_mps /= new_mps.norm()
            # Use warm-built layer with eps-perturbed identity (close to identity
            # so applying it leaves chi_fid unchanged).
            layer = build_identity_layer_with_warm_structure(
                mps_np, chi, eps=0.0, seed=seed,
            )

        layers.append(layer)
        init_losses.append(init_loss)
        final_losses.append(final_loss)
        mps = new_mps.copy()
        mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
        f_chi = chi_fidelity(new_mps, chi)
        chi_fids.append(f_chi)
        if verbose:
            print(f"    [post layer {L+1}] chi={chi} fid = {f_chi:.6f}   "
                  f"max_bond = {new_mps.max_bond()}")

    return {
        'chi':           chi,
        'num_layers':    num_layers,
        'layers':        layers,
        'chi_fids':      chi_fids,
        'init_losses':   init_losses,
        'final_losses':  final_losses,
        'final_mps':     mps,
    }


print('Identity-init helpers loaded: build_identity_layer_with_warm_structure, '
      'optimise_identity_layer, sequential_disentangle_identity_struct.')

