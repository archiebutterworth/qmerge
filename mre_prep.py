# =============================================================================
#  Q-MERGE STAGE 1 -- preparation and SELECT-SWAP compression
# =============================================================================
#
#  WHAT THIS IS
#  ------------
#  The circuit-construction half of Q-MERGE. Given M normalised segments
#  |psi_i> in C^N, it builds the Qiskit circuit that prepares them on separate
#  registers and compresses them onto a shared storage+index register.
#
#  THE PROTOCOL (paper Sec. II A)
#  --------------------------------
#    1. Prepare |psi_i> on preparation register A_i, all in parallel.
#    2. Put storage S and index I into uniform superposition.
#    3. Apply SELECT-SWAP:  sum_i |i><i|_I (x) SWAP(A_i, S)
#    4. Measure every A_i in the computational basis, outcome u.
#
#  This collapses deterministically onto the "weighted stack"
#
#      |psi_w> = (1/sqrt(Norm)) sum_i w_i |psi_i>_S |i>_I ,
#      w_i = prod_{k != i} (psi_k)_{u_k}
#
#  i.e. branch i picks up the product of the u_k-th amplitudes of all the OTHER
#  segments. Stage 1 never fails; the randomness lands entirely in the weights,
#  which Stage 2 then corrects.
#
#  CONVENTIONS THAT MATTER
#  -----------------------
#  Qiskit little-endian within each register, and register order low -> high is
#  S, I, A_0, ..., A_{M-1}, [anc]. Storage+index therefore occupy the lowest
#  (n+m) qubits as a contiguous block, which is what makes
#  full_index = si + prep * 2**(n+m) work. Get this wrong and the smoke tests in
#  03_hardware_circuit_build.ipynb fail loudly -- which is what they are for.
#
#  USED BY
#  -------
#  03_hardware_circuit_build.ipynb   (the 53-qubit H2-1 circuit)
#
# =============================================================================

"""mre_prep.py -- Multi-register CSWAP-compression state preparation (Stage 1).

Implements the Stage-1 "preparation and compression" of the paper *Parallelism of
State Preparation Algorithms via Multi-Register Encoding*: prepare M chunks on
separate registers, then compress them onto a storage+index register via
SELECT-SWAP gates, producing the weighted stack

    |psi_w> = (1/sqrt(N)) sum_i w_i |psi_i>_S |i>_I ,   w_i = prod_{k!=i} (psi_k)_{u_k}

for a computational-basis measurement outcome u=(u_0,...,u_{M-1}) on the prep
registers (paper Eq. 7-9).

Built for Qiskit 2.x + qiskit-aer. Includes an analytic oracle and helpers so a
companion notebook can smoke-test the built circuit by taking the inner product of
the simulated storage+index state with the analytic |psi_w>.

CONVENTIONS (all verified by the smoke tests in the companion notebook)
----------------------------------------------------------------------
* Chunk psi_i is a length-N numpy vector, N = 2**n; amplitude index k is the
  computational basis state |k>. Index i in {0..M-1}, M = 2**m.
* Qiskit little-endian: within any register, qubit 0 is the least-significant bit
  of the amplitude index.
* Register order (low -> high qubit index): S, I, A_0, ..., A_{M-1}, [anc].
  So the STORAGE+INDEX pair occupies the lowest (n+m) qubits, which makes it a
  contiguous low block of the full statevector: full_index = si + prep*2**(n+m).
* Storage/index ordering inside that low block: si = s + i*N  (S low, I high),
  matching the paper's |psi_i>_S (x) |i>_I in Eq. (7).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation, SwapGate


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _n_qubits(dim: int) -> int:
    n = int(round(np.log2(dim)))
    if 2 ** n != dim:
        raise ValueError(f"dimension {dim} is not a power of two")
    return n


def _normalize(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(vec)
    if nrm == 0:
        raise ValueError("cannot normalize a zero vector")
    return vec / nrm


# --------------------------------------------------------------------------- #
# per-chunk state preparation (pluggable)
# --------------------------------------------------------------------------- #
def row_prep_exact(vec) -> QuantumCircuit:
    """Exact prep of a normalized length-N vector on n qubits via StatePreparation.

    This is the *oracle* prep: U|0> = |psi_i> to machine precision, so any residual
    infidelity in the end-to-end smoke test is attributable to the compression
    machinery, not the prep. The real SSO/MPS prep circuits are dropped in later by
    passing a different `prep_builder` to `build_compression_circuit`.
    """
    vec = _normalize(vec)
    n = _n_qubits(len(vec))
    qc = QuantumCircuit(n, name="Uprep")
    qc.append(StatePreparation(vec), range(n))
    return qc


# --------------------------------------------------------------------------- #
# C^m SWAP_n  (SELECT-SWAP of a whole register, controlled on I == i)
# --------------------------------------------------------------------------- #
def _select_swap_logical(qc, I, i, A, S):
    """Reference (undecomposed) SELECT-SWAP: SWAP(A, S) controlled on I == i,
    using an m-controlled SwapGate per qubit pair. No ancilla. Used to cross-check
    the paper's decomposition."""
    m = len(I)
    cswap = SwapGate().control(m, ctrl_state=i)
    for k in range(len(A)):
        qc.append(cswap, [*I, A[k], S[k]])


def _select_swap_decomposed(qc, I, i, A, S, anc):
    """Paper Fig. 2 decomposition of C^m SWAP_n into 2 C^m NOT gates and n Fredkin
    (CSWAP) gates using one zeroed ancilla:

        flag = (I == i)          -- one C^m NOT onto the zeroed ancilla
        for k: CSWAP(flag, A[k], S[k])
        uncompute flag           -- the second C^m NOT restores the ancilla to |0>

    For m index controls, C^m NOT is an MCX with ctrl_state=i (open/closed controls
    realising the bit pattern of i). At m=2 (M=4 chunks) this is just a Toffoli.
    """
    ctrls = list(I)
    qc.mcx(ctrls, anc[0], ctrl_state=i)          # compute flag = (I == i)
    for k in range(len(A)):
        qc.cswap(anc[0], A[k], S[k])             # Fredkin per storage qubit
    qc.mcx(ctrls, anc[0], ctrl_state=i)          # uncompute flag (restore anc -> |0>)


# --------------------------------------------------------------------------- #
# full Stage-1 compression circuit
# --------------------------------------------------------------------------- #
def build_compression_circuit(chunks, prep_builder=row_prep_exact, decomposed=True,
                              prep_circuits=None, barrier=False):
    """Build the Stage-1 CSWAP-compression circuit for a list of M chunk vectors.

    Parameters
    ----------
    chunks : list of M vectors, each length N = 2**n.
    prep_builder : callable vec -> QuantumCircuit(n) preparing |psi_i> on n qubits.
        Default `row_prep_exact`. Ignored if `prep_circuits` is given.
    decomposed : if True use the paper Fig. 2 decomposition (+1 zeroed ancilla);
        if False use the undecomposed m-controlled swaps (for cross-checking).
    prep_circuits : optional list of M pre-built QuantumCircuit(n) prep circuits (e.g. the
        SSO staircase circuits). Composed (not appended) so their gates render inline.
    barrier : if True, insert a barrier between the prep+H stage and the SELECT-SWAP
        compression. Purely a visual/scheduling separator -- a no-op for statevector
        simulation -- that stops the first C^mNOT from sliding left into the prep region
        when drawing. Strip it before hardware compilation (it fences optimisation).

    Returns
    -------
    qc : QuantumCircuit
    layout : dict with registers and sizes (M, N, n, m, S, I, As, anc, n_low, total).
    """
    chunks = [_normalize(c) for c in chunks]
    M = len(chunks)
    N = len(chunks[0])
    if any(len(c) != N for c in chunks):
        raise ValueError("all chunks must have the same length")
    n = _n_qubits(N)
    m = _n_qubits(M)

    S = QuantumRegister(n, "S")
    I = QuantumRegister(m, "I")
    As = [QuantumRegister(n, f"A{i}") for i in range(M)]
    regs = [S, I, *As]
    anc = None
    if decomposed:
        anc = QuantumRegister(1, "anc")
        regs.append(anc)

    qc = QuantumCircuit(*regs)

    # (1) prepare each chunk |psi_i> on its preparation register A_i
    if prep_circuits is not None and len(prep_circuits) != M:
        raise ValueError(f"prep_circuits has {len(prep_circuits)} entries, expected M={M}")
    for i in range(M):
        prep_i = prep_circuits[i] if prep_circuits is not None else prep_builder(chunks[i])
        qc.compose(prep_i, qubits=list(As[i]), inplace=True)

    # (2) put storage and index into uniform superposition
    qc.h(S[:])
    qc.h(I[:])

    if barrier:
        qc.barrier()

    # (3) SELECT-SWAP: for each i, SWAP(A_i, S) controlled on I == i
    for i in range(M):
        if decomposed:
            _select_swap_decomposed(qc, I, i, As[i], S, anc)
        else:
            _select_swap_logical(qc, I, i, As[i], S)

    layout = dict(M=M, N=N, n=n, m=m, S=S, I=I, As=As, anc=anc,
                  n_low=n + m, total=qc.num_qubits)
    return qc, layout


# --------------------------------------------------------------------------- #
# analytic oracle + statevector extraction (for smoke tests)
# --------------------------------------------------------------------------- #
def weighted_stack_vector(chunks, u) -> np.ndarray:
    """Analytic |psi_w> (length M*N, normalized) for measurement outcome u, in the
    circuit's S(low) (x) I(high) ordering: amp[i*N + s] = w_i * psi_i[s],
    w_i = prod_{k!=i} psi_k[u_k]  (paper Eq. 9)."""
    chunks = [_normalize(c) for c in chunks]
    M = len(chunks)
    N = len(chunks[0])
    out = np.zeros(M * N, dtype=complex)
    for i in range(M):
        w_i = np.prod([chunks[k][u[k]] for k in range(M) if k != i])
        out[i * N:(i + 1) * N] = w_i * chunks[i]
    return _normalize(out)


def outcome_probability(chunks, u) -> float:
    """Born probability P(u) of the joint prep-register outcome u (paper Eq. B1):
    P(u) = (1/(M N)) sum_i prod_{k!=i} |psi_k[u_k]|^2."""
    chunks = [_normalize(c) for c in chunks]
    M = len(chunks)
    N = len(chunks[0])
    tot = 0.0
    for i in range(M):
        tot += float(np.prod([abs(chunks[k][u[k]]) ** 2 for k in range(M) if k != i]))
    return tot / (M * N)


def sample_outcomes(chunks, n_samples=1, seed=None):
    """Efficient O(M log N) sampler of the compression-measurement outcome u
    (paper Appendix B). P is a uniform mixture of M distributions Q_i; under Q_i
    register i is uniform on {0..N-1} and every other register k follows its Born
    distribution |psi_k|^2. Returns a list of length-M tuples."""
    rng = np.random.default_rng(seed)
    chunks = [_normalize(c) for c in chunks]
    M, N = len(chunks), len(chunks[0])
    probs = [np.abs(c) ** 2 for c in chunks]
    probs = [p / p.sum() for p in probs]
    out = []
    for _ in range(n_samples):
        i = int(rng.integers(M))
        u = [int(rng.integers(N)) if k == i else int(rng.choice(N, p=probs[k]))
             for k in range(M)]
        out.append(tuple(u))
    return out


def peak_outcome(chunks):
    """A deterministic high-support outcome: u_k = argmax_k |psi_k|. Guarantees every
    factor (psi_k)_{u_k} is nonzero, so the reconstruction is well-posed."""
    return tuple(int(np.argmax(np.abs(_normalize(c)))) for c in chunks)


def extract_SI(full_sv, u, layout):
    """Project the prep registers onto |u> (and ancilla onto |0>) and return the
    (normalized S(x)I block, block_norm).

    Because S,I are the lowest n+m qubits and the prep registers (+anc) are the high
    qubits, the outcome-u / anc-0 amplitudes are the contiguous slice
        full_sv[high*(N*M) : (high+1)*(N*M)],  high = sum_i u_i * N**i.
    block_norm**2 is the probability of that joint outcome; it should match
    outcome_probability(chunks, u).
    """
    full_sv = np.asarray(full_sv).reshape(-1)
    M, N, n_low = layout["M"], layout["N"], layout["n_low"]
    high = sum(int(u[i]) * (N ** i) for i in range(M))
    width = 2 ** n_low  # = N*M
    block = full_sv[high * width:(high + 1) * width]
    nrm = np.linalg.norm(block)
    normed = block / nrm if nrm > 0 else block
    return normed, float(nrm)


def ancilla_leak(full_sv, layout) -> float:
    """Norm of the amplitude with the zeroed ancilla in |1> -- should be ~0 if the
    C^m NOT flags are correctly uncomputed. Returns 0.0 if there is no ancilla."""
    if layout["anc"] is None:
        return 0.0
    full_sv = np.asarray(full_sv).reshape(-1)
    half = len(full_sv) // 2       # ancilla is the highest qubit
    return float(np.linalg.norm(full_sv[half:]))


def fidelity(a, b) -> float:
    """|<a|b>|^2 for two (not necessarily normalized) vectors."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) ** 2)


# --------------------------------------------------------------------------- #
# statevector simulation
# --------------------------------------------------------------------------- #
def simulate_statevector(qc) -> np.ndarray:
    """Full statevector of `qc` via Aer (method='statevector'). Returns a complex
    numpy array indexed in Qiskit little-endian (qubit 0 = LSB)."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector")
    tqc = qc.copy()
    tqc.save_statevector()
    result = sim.run(transpile(tqc, sim)).result()
    return np.asarray(result.get_statevector())
