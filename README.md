# Q-MERGE: Parallelising Quantum State Preparation for Large-Scale Classical Data

Code and data accompanying the paper of the same name.

**Archie Butterworth**, **Jens Renders**, **Jingbo Wang**
Centre for Quantum Information, Simulation and Algorithms, The University of Western
Australia — and University Paris City / University of Reunion (J.R.)

Q-MERGE partitions a large target state into *M* *n*-qubit segments, prepares them
independently and in parallel, and merges them onto a single register using
SELECT-SWAP operations and measurement. Stage 1 produces a *weighted stack* with
guaranteed success; Stage 2 restores the original segment weights through a
block-encoded diagonal operator, with a classical preconditioning step that makes the
post-selection succeed often enough to be practical.

Every figure in the paper that comes from code can be regenerated from what is in
here. Long-running results (SSO sweeps that take CPU-hours, and all Quantinuum
hardware shots, which cannot be re-run without machine time) are committed as raw
data files so the plotting notebook runs immediately.

---

## Quick start

```bash
pip install -r requirements.txt
jupyter lab 05_figures.ipynb
```

Run the notebooks **with the repository root as the working directory** — they
resolve all paths relative to `os.getcwd()`.

Developed against Python 3.11. `requirements.txt` pins the exact versions used for the
published results; see the note at the top of that file about optimiser drift.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_qmerge_engine.ipynb` | The compute engine: per-segment SSO row sweeps, direct full-state encoding baselines, brickwall preconditioning, and the Stage-2 reweighting success-probability sweeps. This is what writes `SSO results/` and `results/`. Hours of CPU time. |
| `02_haar_scaling_sweep.ipynb` | Reweighting success probability *P* vs *M* and *N* for Haar-random segments → `sweep_success_prob_*.csv` |
| `03_hardware_circuit_build.ipynb` | Builds the 53-qubit Stage-1 circuit (4 preparation registers × 10 qubits, 10-qubit storage, 2 index qubits, 1 ancilla) and exports the `.qpy` artefacts. Despite the filename note inside, this is the 4 × 1024 run. |
| `04_shadow_overlap_and_calibration.ipynb` | Per-branch shadow-overlap certification on H2-1 (needs an account) **and** the per-branch calibration curves, 9000 points each (runs anywhere) |
| `05_figures.ipynb` | **Produces Figures 3, 4 and 5, and Table I** |

Supporting modules, imported directly by the notebooks:

| Module | Role |
|---|---|
| `mre_prep.py` | Q-MERGE Stage 1: SELECT-SWAP compression circuits (Qiskit) |
| `sso_prep.py` | The SSO / MPS staircase state-preparation stack |
| `sso_rowprep.py` | Wraps `sso_prep` into Qiskit staircase row-prep circuits |
| `sso_framework_setup.py` | Earlier standalone copy of the SSO framework, used by `02_haar_scaling_sweep.ipynb` |

## Data

| Path | Contents |
|---|---|
| `R75 Ultrasound data.xlsx` | Raw ultrasound data: 382 samples × 128 detector elements. The analysis resamples the sample axis to *N* = 1024 (amplitude encoding needs a power of two) and takes the middle *M* = 32 of the 128 registers, giving the 32 × 1024 target state. |
| `SSO results/row_sweep_*/` | Per-segment SSO fidelities vs staircase depth (`row_fids.json`, `row_states.npz`, `sigma.npy`) |
| `SSO results/fullstate_*/` | Direct whole-state encoding baselines (`checkpoint.npz`) |
| `results/v_R75x4_M32_incr.pkl` | 5000 simulated Stage-1 measurement outcomes, baseline and preconditioned (Figure 3b) |
| `sweep_success_prob_*.csv`, `precond_vs_M.csv` | Haar-random scaling sweeps (Figure 5) |
| `shadow_branch_hw_4x1024_L1_mid.pkl` | Branch-resolved H2-1 shots, 450 per branch (150 each of *X*, *Y*, *Z*) |
| `shadow_vs_fidelity_rows_4x1024_L1_mid_vs_ideal.npz` | Noiseless ω-vs-fidelity calibration curve, 9000 points per branch |
| `mre_{qc,prep_circuits,target_data}_4x1024_L1_mid.*` | The submitted circuits and their target segments |
| `*_4x256_L1_mid.*`, `shadow_branch_hw.pkl`, `mre_qc.qpy`, … | Earlier, smaller hardware runs (4 × 256 and 4 × 32). No figure in the paper uses them, but notebooks `03` and `04` still reference them, so they are kept. |

Final PDFs of the three published data figures are in `figures/`.

---

## Figure provenance

| Paper figure | Notebook (cell) | Principal inputs |
|---|---|---|
| 3 — dataset, reweighting success, infidelity vs depth and vs CPU time | `05_figures.ipynb` (2) | `R75 Ultrasound data.xlsx`, `SSO results/row_sweep_*`, `SSO results/fullstate_R75x4_M32`, `results/v_R75x4_M32_incr.pkl` |
| 4 + Table I — per-branch H2-1 shadow overlap | `05_figures.ipynb` (4) | `shadow_branch_hw_4x1024_L1_mid.pkl`, `shadow_vs_fidelity_rows_4x1024_L1_mid_vs_ideal.npz`, `mre_prep_circuits_4x1024_L1_mid.qpy` |
| 5 — *P* vs *M* for Haar-random segments | `05_figures.ipynb` (6) | `sweep_success_prob_vs_M.csv` |

The calibration curves — the (ω, *F*) scatters of Appendix F that convert a measured
shadow overlap into a fidelity — are generated by
`04_shadow_overlap_and_calibration.ipynb` (cell 79 for the 4 × 1024 branches, cell 68
for the earlier 4 × 32 run). Both run without a Quantinuum account.

Figures 1 and 2 (the Q-MERGE Stage-1 circuit diagrams) are drawn in LaTeX in the paper
source and have no code here.

## Reproducing the headline numbers

| Result | Where |
|---|---|
| Q-MERGE infidelity 3.5 × 10⁻⁹ vs 6.2 × 10⁻³ direct, at matched CNOT depth | `05_figures.ipynb`, cell 2 |
| Preconditioning raises *P* from 0.0098 to 0.3170 for 18 CNOT of depth | `05_figures.ipynb`, cell 2 |
| Per-branch fidelities 0.589, 0.790, 0.721, 0.720 (Table I) | `05_figures.ipynb`, cell 4 |
| Weighted branch fidelity 0.7135 ± 0.0720 on H2-1 | `05_figures.ipynb`, cell 4 |
| *P* > 5 % out to *M* = 10⁷ | `05_figures.ipynb`, cell 6 |

## Notes

- **No credentials are stored here.** The notebook that submits jobs to H2-1 reads
  `QNX_USER` and `QNX_PASS` from the environment:

  ```bash
  export QNX_USER="you@example.com"
  export QNX_PASS="..."
  ```

  Only the submission cells need them. The calibration simulations, all analysis and
  all plotting run offline from the committed result files, so **every figure in the
  paper reproduces without a Quantinuum account.**
- Notebooks are committed **with their outputs** so the results are readable on GitHub
  without running anything.
- Every notebook opens with a header cell explaining what it does, what it consumes
  and what it writes. Cell numbers quoted in this README and in those headers are
  0-indexed and include that header cell.
- Paths are resolved relative to the working directory. A few comments and committed
  cell outputs still mention the original `C:\qruns\...` and OneDrive scratch
  locations; those checkpoints now live under `SSO results/`.
- Re-running a figure cell writes its PDF **next to the notebook**, not into
  `figures/`. The `figures/` directory holds the versions used in the paper, so
  regenerating never silently overwrites them.

## Related work

1. A. Butterworth, J. Green, Y. Wu, J. Pan, J. Wang, *Efficient quantum state
   preparation on Quantinuum hardware*, arXiv [quant-ph] (2026). The IPR-minimisation
   result that the classical preconditioning step builds on.
2. J. Green and J. Wang, *Phys. Rev. A* **113**, 052616 (2026). Schmidt Spectrum
   Optimisation (SSO), the segment-level preparation method used throughout.

## Acknowledgements

Quantinuum provided access to the H2-1 trapped-ion quantum computer and recognised our
team as a finalist in their 2026 SG Grand Challenge. This research was funded by the
Australian Government through the Advanced Strategic Capabilities Accelerator (ASCA)
program. The SSO method is due to Green, Snow and Wang; `mps-to-circuit` is used under
its own licence.

## A note on how this repository was assembled

The research, methods, implementation and results presented here are my own. The
source files were written over the course of the project and lived across several
working directories.

Claude Code was used to assemble them into this repository: collecting the relevant
notebooks, modules and data files, rewriting hardcoded absolute paths so the notebooks
run from a clone, removing credentials, and drafting this README and the explanatory
header cells at the top of each notebook.
