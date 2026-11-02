# Data and plotting scripts for "Gutzwiller Quantum-Classical Embedding for Dynamics Simulations of Correlated Electron Systems"

J. Aftergood, D. Guerci, J. Brunton, F. Zhang, T. Iadecola, P. P. Orth, C.-Z. Wang, N. Lanatà, and Y.-X. Yao

This repository contains the data shown in every figure of the paper, plus a script that regenerates each figure from that data.

- Paper: *journal reference and DOI to be added*
- Dataset DOI: [10.5281/zenodo.23887709](https://doi.org/10.5281/zenodo.23887709)

## Quick start

```bash
git clone <this repository> && cd tgqce_figure_data
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./make_figures.sh
```

Each script writes its figure into its own folder: `fig1_schematic/Fig1.pdf`, `fig2_B1_ibmq/Fig2.pdf`, `fig3_B357_avqds/Fig3.pdf` and `fig4_measurement/Fig4.pdf`. The PDFs included in the repository are the versions used in the paper. Regenerating the figures needs only the small processed data files; the raw results in `fig4_measurement/raw/` (23 MB) and `measurement_scripts/` are for recomputing Fig. 4. You can also run a single script directly, e.g. `python fig3_B357_avqds/plot_fig3.py`; the scripts find their data relative to their own location, so they can be run from any directory.

### Requirements

- **Python ≥ 3.9** with the packages in `requirements.txt`:
  - numpy and matplotlib for all figures;
  - QuTiP for the Bloch sphere in Fig. 2.

  Tested with Python 3.11, numpy 2.4, matplotlib 3.11 and QuTiP 5.3.
- **A LaTeX installation** with `latex` and `dvipng` on `PATH`. Figs. 2–4 render text with LaTeX (`text.usetex = True`). Fig. 1 does not need LaTeX.
- **An SVG-to-PDF converter**, needed only for the PDF of Fig. 1. The Fig. 1 script writes an SVG, and `make_figures.sh` converts it with `rsvg-convert` (package `librsvg2-bin` or `librsvg`). Any of these works instead:
  - `inkscape tGQCE_schematic.svg --export-filename=Fig1.pdf`
  - `cairosvg tGQCE_schematic.svg -o Fig1.pdf`

## Physical setting

The data concern interaction quenches of the half-filled single-band Hubbard model on the Bethe lattice, with a semicircular density of states of bandwidth W = 1. The model is solved with the time-dependent ghost Gutzwiller approximation, using B bath orbitals per spin, of which B − 1 are ghost orbitals.

- **Observable:** the double occupancy ⟨↑↓⟩ = ⟨n↑ n↓⟩ of the correlated orbital.
- **Time:** in units of ħ/W.
- **Embedding states (Figs. 3 and 4):** propagated either exactly or with the adaptive variational quantum dynamics simulation (AVQDS).

## Contents

### `fig1_schematic/` (Fig. 1, schematic of the t-GQCE framework)

| file | content |
|---|---|
| `generate_tGQCE_schematic.py` | Draws the schematic as a vector SVG (`python generate_tGQCE_schematic.py`). Mathematical labels are rendered with matplotlib MathText. |
| `tGQCE_schematic.svg`, `Fig1.pdf` | Output: the SVG, and the PDF converted from it. |

### `fig2_B1_ibmq/` (Fig. 2, B = 1: conventional time-dependent Gutzwiller approximation)

For B = 1 the embedding state reduces to a single qubit, |Φ⟩ = c1|0⟩ + c2|1⟩. The hardware runs use two physical qubits: one for the embedding state and one ancilla for the Hadamard-test measurements of the McLachlan matrix M and vector V. Each measurement uses 2 × 10⁴ shots.

| file | columns | content |
|---|---|---|
| `a_double_occupancy_Uf_half_Uc_dyn.dat` | t, ⟨↑↓⟩ | Panel (a): noiseless dynamics after the quench U: 0 → U_c^dyn/2. |
| `b_period_vs_Uf.dat` | U_f, T | Panel (b): oscillation period of ⟨↑↓⟩ versus the final interaction. It diverges at U_c^dyn = U_c^BR/2 = 16/(3π). |
| `c_statevector.csv` | t, ⟨↑↓⟩ | Panel (c), U_f = U_c^dyn: noiseless statevector simulation. |
| `c_noise_model.csv` | t, ⟨↑↓⟩ | Panel (c): simulation with the IBMQ_Quito noise model and readout-error mitigation (inversion of the confusion matrix). |
| `c_ibmq_quito.csv` | t, ⟨↑↓⟩ | Panel (c): run on the IBMQ_Quito device, without error mitigation. |
| `c_statevector_wavefunction.dat` | t, Re c1, Im c1, Re c2, Im c2 | Noiseless embedding state, used for the Bloch-sphere inset. The script plots every 50th of the first 700 points. |
| `plot_fig2.py` | | Writes `Fig2.pdf`. |

The csv files have one header line; their first column is t and their second is ⟨↑↓⟩. The McLachlan time step is 10⁻³.

### `fig3_B357_avqds/` (Fig. 3, B = 3, 5, 7: exact embedding dynamics versus AVQDS)

Two quenches are shown: U: 0.05 → 1.5 (weak) and 0.05 → 2.5 (strong). The DMFT reference is for U: 0 → 1.5 and 0 → 2.5.

| file | columns | content |
|---|---|---|
| `dt_u15_B{1,3,5,7}.dat`, `dt_u25_B{1,3,5,7}.dat` | t, ⟨↑↓⟩ | td-gGA with the embedding states propagated exactly, for U_f = 1.5 and 2.5. |
| `dt_u15_tdmft.dat`, `dt_u25_tdmft.dat` | t, ⟨↑↓⟩ | Nonequilibrium DMFT reference, from M. Eckstein, M. Kollar and P. Werner, Phys. Rev. Lett. 103, 056403 (2009). |
| `avqds_data/B{3,5,7}U{1.5,2.5}.dat` | see below | AVQDS results along the dynamics. |
| `plot_fig3.py` | | Writes `Fig3.pdf`. |

Columns of `avqds_data/*.dat`:
1. `t`.
2. `d_AVQDS`: ⟨↑↓⟩ in the AVQDS state.
3. `N_cx`: CNOT count of the AVQDS circuit. Each Pauli-string generator of weight w counts as 2(w − 1) CNOTs.
4. `n_gates`: number of generators in the ansatz.
5. `infidelity`: 1 − F of the AVQDS state relative to the exactly propagated embedding state.
6. `S_imp`: entanglement entropy between the impurity and bath orbitals of the exact state. It is fixed by ⟨↑↓⟩ and is not plotted.
7. `S_updn`: entanglement entropy between the spin-up and spin-down spin-orbitals of the exact state, shown in the insets.

### `fig4_measurement/` (Fig. 4, measuring the embedding observables, B = 5, U_f = 1.5)

The circuits are the AVQDS states compressed into brickwork circuits at t = 0.5, 1, …, 5 and compiled to a trapped-ion gate set. The double occupancy and the single-particle density matrix ρ are measured with 11 orbital-rotation settings plus a mirror circuit, using 10⁵ shots per time. They are simulated under the noise model of I.-C. Chen et al., npj Quantum Inf. 12, 102 (2026), with all error rates scaled by λ.

| file | content |
|---|---|
| `hw_data_lambda1.dat` | λ = 1, the error rates of the noise model. |
| `hw_data_lambda1over3.dat` | λ = 1/3, three times lower error rates. |
| `plot_fig4.py` | Writes `Fig4.pdf`. Also reads `../fig3_B357_avqds/avqds_data/B5U1.5.dat` for the AVQDS reference curve and CNOT count. |
| `build_hw_data.py` | Rebuilds the two `hw_data_*.dat` files from `raw/` (`python build_hw_data.py`; `--shots 300000` or `1000000` for the larger shot numbers). |
| `raw/` | Raw results of the measurement simulations, described below. |

Each row of `hw_data_*.dat` is one time. The header line names every column:
- `t`, `Ncx_AVQDS`, `Ncx_compressed`, `NZZ_compressed` (two-qubit ZZ gates after compilation), `d_AVQDS`.
- Then, for each of the three cases `shot` (shot noise only), `noisy` (noise model with post-selection) and `mitigated` (with mirror-circuit mitigation):
  - the median, 5% and 95% quantiles over 500 repetitions of the estimated ⟨↑↓⟩ (`*_d_*`);
  - the same three statistics of max_ab |ρ_ab − ρ_ab^AVQDS| (`*_rho_err_*`);
  - the same three statistics of |⟨↑↓⟩ − ⟨↑↓⟩_AVQDS| (`*_d_err_*`).

#### `fig4_measurement/raw/` (raw results of the measurement simulations)

Times t = 0.50, 1.00, …, 5.00 appear in the file names with two decimals; one time is stored as `3.01`, the AVQDS time step closest to 3. The tag `rdm1npj1` stands for λ = 1 and `rdm1npj033` for λ = 1/3.

| file | content |
|---|---|
| `swp/swp_t<t>.json` | Compression of the AVQDS state into brickwork circuits by environment sweeps (`sweep_compress.py`). Holds one entry per depth: fidelity, ⟨↑↓⟩ and ρ errors, and CNOT count. |
| `swp/swp_t<t>.qpy` | The compressed circuits for all depths, in Qiskit QPY format, named `L<depth>`. Load them with `qiskit.qpy.load`. |
| `<tag>_t<t>.json` | Noise-model results without shot noise (`rdm1.py`). Holds the chosen circuit, the 11 measurement settings, post-selection acceptance, mirror-circuit data, and ρ and ⟨↑↓⟩ raw, post-selected and mitigated, compared with the AVQDS values (`rho_avqds`, `d_avqds`). |
| `<tag>_probs_quantinuum_t<t>.npz` | Exact outcome distributions of all settings and of the mirror circuit under the noise model, from which the finite-shot samples are drawn. |
| `budget_sym_<tag>.json` | Finite-shot statistics (`budget.py --sym --w-id 3`): 500 repetitions at 10⁵, 3 × 10⁵ and 10⁶ shots per time for the three cases. Also holds the infinite-shot values. |

Every element ρ_ab = ⟨a_a† a_b⟩ refers to the modes of one spin block. The qubit order is bath ↑ (0–4), impurity ↑ (5), impurity ↓ (6), bath ↓ (7–11), under the Jordan–Wigner encoding with an occupied mode mapped to |1⟩.

### `measurement_scripts/` (measurement-simulation code)

These are the Python scripts used for Fig. 4 and for the comparison of measurement strategies in Sec. VI. Their dependencies are pinned in `measurement_scripts/requirements.txt`: Qiskit, Qiskit Aer, pytket and quimb. Each script has a full description and usage line in its docstring. The chain for Fig. 4 is:

1. `sweep_compress.py RESULTS_ASIM.h5 swp/swp --times ...`: compresses the AVQDS states into brickwork circuits. It writes `swp/swp_t<t>.{json,qpy}`.
2. `rdm1.py DIR --times ... --noise npj --lam 1 --pmem 0 --out rdm1npj1`, and the same with `--lam 0.333333 --out rdm1npj033`. This compiles the circuits to the trapped-ion gate set (`hw_compile.py`), adds the 11 orbital-rotation settings and the mirror circuit, and simulates them under the noise model. It writes `<tag>_t<t>.json` and `<tag>_probs_quantinuum_t<t>.npz`. The circuit at each time is the one selected in the paper's analysis; `rdm1.py` normally reads that choice from earlier output files via `--pin`.
3. `budget.py DIR --inp <tag> --sym --w-id 3 --shots 100000 300000 1000000`: draws the finite-shot samples (seed 1), applies symmetry averaging and mitigation, and writes `budget_sym_<tag>.json`.
4. `fig4_measurement/build_hw_data.py`, then `plot_fig4.py`.

Steps 3 and 4 run entirely from the files in `raw/`. For example:

```bash
pip install -r measurement_scripts/requirements.txt
python measurement_scripts/budget.py fig4_measurement/raw --inp rdm1npj033 --sym --w-id 3 --shots 100000 300000 1000000
```

This takes about 4 minutes on one core and reproduces `raw/budget_sym_rdm1npj033.json` exactly. Steps 1 and 2 also need the AVQDS states (`results_asim.h5`, tens of MB per run), which are not included. `hw_compile.py` looks for them at `../AVQDS/<case>/results_asim.h5` relative to the scripts.

The other scripts are as follows:
- `rdm1_shots.py` and `rdm1_fshadow.py`: the comparison of measurement strategies, covering qubit-wise commuting Pauli groups, Pauli shadows and fermionic shadows.
- `aqc_compress.py` and `mps_compress.py`: alternative compression schemes that were tried.
- `rho_compression.py` and `summarize_*.py`: helpers.

The scripts default to an earlier noise model, with depolarizing errors of 3 × 10⁻⁵ and 10⁻³ and memory errors. The paper uses `--noise npj` throughout.

## How the data were produced

Apart from `fig4_measurement/raw/`, these are processed data: each file holds exactly the quantities plotted in its figure. They were produced as follows:

- **td-gGA dynamics with exact embedding states:** a Fortran code integrating the td-gGA equations of motion (RK45, relative tolerance 10⁻⁷).
- **AVQDS and AVQITE:** statevector simulations with the adaptive variational codes of the authors' group. The McLachlan-distance threshold is 10⁻⁴.
- **Fig. 2, panel (c):** Qiskit simulations, and runs on the IBMQ_Quito device.
- **Fig. 4:** density-matrix simulations of the compiled circuits with Qiskit Aer, followed by finite-shot sampling. The scripts and raw results are included, as described above.

The td-gGA and AVQDS codes are not part of this repository; questions about them can be addressed to the corresponding author.

## License

The data files (`*.dat`, `*.csv`, `*.svg`, `*.pdf`) are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and the scripts (`*.py`, `*.sh`) under the MIT License; see `LICENSE`.

## Citation

If you use these data, please cite the paper above.
