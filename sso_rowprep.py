# =============================================================================
#  SSO -> QISKIT ADAPTER -- segment preparation circuits for Q-MERGE
# =============================================================================
#
#  WHAT THIS IS
#  ------------
#  The bridge between the two halves of the pipeline. sso_prep works in
#  quimb tensor networks; mre_prep needs Qiskit circuits. This module runs SSO on
#  a segment and emits a QuantumCircuit whose statevector equals that segment, to
#  the SSO fidelity, in the SAME little-endian convention Qiskit's
#  StatePreparation uses -- so the result drops straight into
#  mre_prep.build_compression_circuit as a prep circuit.
#
#  THE CONVENTION (grid-searched against the row statevector, not guessed)
#  ----------------------------------------------------------------------
#    site i            ->  qiskit qubit (n - 1 - i)
#    gates             ->  forward qubit order
#    prep gates        ->  as returned by mps_to_circuit_quimb_warm
#    layer gates       ->  tensor.reshape(N, N).conj(), applied in REVERSE order
#                          after the analytic prep of the chi-truncated residual
#
#  WHY qc.unitary() AND NOT StatePreparation
#  -----------------------------------------
#  Gates are added with qc.unitary(...) so that when this circuit is composed into
#  the multi-register circuit, the individual two-qubit staircase gates stay
#  visible and get compiled individually. StatePreparation would collapse them
#  into one opaque instruction and inflate the compiled gate count.
#
#  USED BY
#  -------
#  03_hardware_circuit_build.ipynb
#
# =============================================================================

"""sso_rowprep.py -- Qiskit staircase row-prep circuits from the SSO/MPS stack.

Wraps `sso_prep` (the extracted disentangler + analytic MPS->circuit code) and emits a
Qiskit QuantumCircuit whose statevector equals the target row to the SSO fidelity, in
the SAME little-endian convention as `qiskit` `StatePreparation` -- so it drops directly
into `mre_prep.build_compression_circuit` as a prep circuit.

Convention (grid-searched against the row statevector, see sso_probe3):
  site i  ->  qiskit qubit (n-1-i);  gates in forward-qubit order;
  prep gates as returned by `mps_to_circuit_quimb_warm`;
  layer gates = tensor.reshape(N,N).conj(), layers applied in REVERSE order after the
  analytic prep of the chi-truncated final residual.
Gates are added with `qc.unitary(...)` (not `StatePreparation`) so that composing the
circuit into the multi-register circuit renders the individual 2-qubit staircase gates.
"""

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import sso_prep as sso


def _norm(v):
    v = np.asarray(v, dtype=complex).reshape(-1)
    return v / np.linalg.norm(v)


def _fid(a, b):
    a, b = _norm(a), _norm(b)
    return float(abs(np.vdot(a, b)) ** 2)


def _assemble_circuit(layers, gate_infos, final_ct, n, label_prep, label_V):
    """Assemble the staircase Qiskit circuit from the analytic residual prep + reverse layers."""
    q = lambda qubits: [n - 1 - x for x in qubits]        # site -> qubit (verified convention)
    _, prep_gi = sso.mps_to_circuit_quimb_warm(final_ct)
    qc = QuantumCircuit(n, name="Uprep_SSO")
    for _, qubits, U in prep_gi:                          # (1) prep of chi-truncated residual
        qc.unitary(np.asarray(U), q(qubits), label=label_prep)
    for L in range(len(layers) - 1, -1, -1):             # (2) staircase V-layers, reversed
        gi = gate_infos[L]
        layer = layers[L]
        for i, (_, qubits, _) in enumerate(gi):
            t = layer.tensors[i]
            N = 2 ** (len(t.inds) // 2)
            arr = t.data.detach().cpu().numpy() if hasattr(t.data, "detach") else np.asarray(t.data)
            qc.unitary(arr.reshape(N, N).conj(), q(qubits), label=label_V)
    return qc


def build_row_circuit(vec, chi=2, num_layers=10, n_iter=200, label_prep="Vp", label_V="V",
                      max_tries=6, verify_tol=1e-6, degeneracy_eps=1e-10, verbose=False):
    """QuantumCircuit(n) staircase preparing `vec` (Statevector(qc) ~= vec, little-endian).

    KNOWN FAILURE MODE (diagnosed, not cosmetic): for some real-valued rows, the chi=2
    disentangler drives a residual to an EXACT Schmidt-rank-1 cut at some interior bond (not
    just approximately low-rank -- exactly degenerate). `mps_to_circuit_quimb_warm` assumes
    every interior bond has dimension exactly chi and emits a wrong-arity gate there, which
    crashes the assembly with a matmul shape mismatch. Whether a given fit lands exactly on
    this degeneracy is NOT reliably pinned down by `torch.manual_seed` alone -- PyTorch's
    multi-threaded CPU ops are not bit-reproducible run-to-run even with a fixed seed, so the
    optimiser's converged parameters differ at the float-ULP level between runs, which is
    normally harmless but occasionally lands exactly on (or just off) the degenerate point.
    What IS confirmed deterministic: replaying an ALREADY-FIXED (mps, layers) pair via
    `img_sso_replay` reproduces the same crash every time (`from_dense`'s SVD sweep is
    deterministic given its input) -- so retrying only the replay step, on unchanged layers,
    never helps (confirmed: 15/15 identical failures). Measured single-attempt failure rate
    ~15-20% across a seed sweep on this dataset, so it is not a rare edge case to ignore.

    Fix: on retry, refit from scratch against a copy of `vec` perturbed by a fixed, tiny,
    reproducible complex Gaussian (relative magnitude `degeneracy_eps`, seeded by the attempt
    index) -- enough to break the exact algebraic degeneracy regardless of which upstream
    mechanism produced it. Fidelity cost of the perturbation itself is O(degeneracy_eps^2) ~
    1e-20 at the default, unmeasurable next to the SSO floor (~1e-7 to 1e-9) -- confirmed
    empirically to resolve every reproduced failure at degeneracy_eps as small as 1e-10.
    `circuit_fid` is always reported against the TRUE, unperturbed `vec`, so retried rows are
    not silently graded on an easier target.

    Assembly correctness (independent of fit quality -- chi=2 on a state with more
    entanglement than num_layers can remove will legitimately plateau below fidelity 1; that
    is `sso_fid`, not an error) is checked against the SSO's own dense reconstruction
    (`img_sso_reconstruct`), which is what the circuit is supposed to equal by construction.

    `qc.metadata`:
      'sso_fid'     -- final chi_fid reported by the disentangler sweep (fit quality).
      'circuit_fid' -- fidelity of the built circuit against the TRUE, unperturbed `vec`
                       (== sso_fid up to float error; this is the number to report/plot).
      'assembly_fid'-- fidelity of the built circuit against the SSO's dense reconstruction
                       (the retry gate; should be ~1.0 whenever an attempt is accepted).
      'perturbed'   -- True if this row needed the degeneracy-breaking perturbation.
    """
    row = _norm(vec)
    n = int(round(np.log2(len(row))))

    last = None
    for attempt in range(max_tries):
        try:
            if attempt == 0:
                fit_target = row
            else:                                          # break the exact degeneracy
                rng = np.random.default_rng(1000 + attempt)
                noise = rng.standard_normal(len(row)) + 1j * rng.standard_normal(len(row))
                fit_target = _norm(row + degeneracy_eps * noise)

            mps = qtn.MatrixProductState.from_dense(fit_target.astype(np.complex128),
                                                     dims=[2] * n, cutoff=0.0)
            mps /= mps.norm()
            res = sso.sequential_disentangle_warm(mps, chi=chi, num_layers=num_layers,
                                                   n_iter=n_iter, verbose=verbose)
            layers, chi_fids = res["layers"], res["chi_fids"]

            replay = sso.img_sso_replay(mps, layers, chi)
            gate_infos = replay["gate_infos"]
            final_state = _norm(replay["residual_states"][-1])
            final_ct = qtn.MatrixProductState.from_dense(final_state, dims=[2] * n, cutoff=0.0)
            final_ct /= final_ct.norm()
            final_ct.compress(max_bond=chi, cutoff=0.0)
            final_ct /= final_ct.norm()
            qc = _assemble_circuit(layers, gate_infos, final_ct, n, label_prep, label_V)

            dense_recon = sso.img_sso_reconstruct(mps, layers, gate_infos, chi, K=len(layers))
            f_asm = _fid(np.asarray(Statevector(qc)), dense_recon)
            if f_asm > 1 - verify_tol:
                f_target = _fid(np.asarray(Statevector(qc)), row)          # vs TRUE row
                qc.metadata = {"sso_fid": float(chi_fids[-1]), "circuit_fid": f_target,
                               "assembly_fid": f_asm, "num_layers": num_layers, "chi": chi,
                               "tries": attempt + 1, "perturbed": attempt > 0,
                               "chi_fids": [float(x) for x in chi_fids]}
                return qc
            last = f"assembly fidelity {f_asm:.3e} below tol on attempt {attempt+1}"
        except Exception as e:                     # exact-degeneracy assembly crash -> perturb & refit
            last = f"{type(e).__name__}: {e} (attempt {attempt+1})"
        if verbose:
            print(f"    retry (refit, perturbed): {last}")
    raise RuntimeError(f"build_row_circuit failed after {max_tries} tries -- last: {last}")


def build_prep_circuits(chunks, chi=2, num_layers=10, n_iter=200, verbose=True):
    """Fit every chunk; return (list_of_circuits, list_of_final_fids)."""
    circuits, fids = [], []
    for i, c in enumerate(chunks):
        qc = build_row_circuit(c, chi=chi, num_layers=num_layers, n_iter=n_iter)
        f = qc.metadata["sso_fid"]
        circuits.append(qc)
        fids.append(f)
        if verbose:
            print(f"  row {i}: SSO final fid = {f:.10f}   (infidelity {1 - f:.2e})")
    return circuits, fids
