# 01 — The project: what it is, why, and where it leads

What this project must achieve and what it is deliberately not; status lives in `02_STATUS.md`
Part A. A orientation and working rules · B mandate, scope, criteria, ladder, hardware · C the
self-interaction programme · D the role of ML · E from corpus to simulation · F publication ·
G open problems · H binding specification answers. Part letters and section numbers are addresses
cited from the code and the other documents; a gap in the numbering is a folded section, never a
lost one.

---

## Part A — Orientation

### 1. What this project is — and is not

An instrument and a dataset: a real-space, GPU-capable ground-state KS solver for isolated molecules
on a uniform grid, the nuclear cusp factored out analytically on the all-electron path, producing
gate-certified densities, potentials and energies plus diagnostics of where standard functionals
fail. Its target is self-interaction / flat-plane error (D-22). **It contains no machine learning**;
"classical" means *non-neural*, not classical-fluid DFT ([A17], Brémond et al., *Contemporary DFT*,
PCCP 2026, DOI 10.1039/D5CP03373J). `src/cdft/` holds no training loop, network or optimiser, and
gate **G5.7** enforces that by static import scan. The boundary is Part B §3.4.

### 2. Where things are

Six documents in `docs/`, one purpose each (D-73).

| File | Purpose |
|---|---|
| `00_LITERATURE_SURVEY.md` | the tagged references `[A1]`… with what is taken from each; the ML-XC map is section L |
| `01_PROJECT.md` (this file) | what and why: mandate, boundary, criteria, ladder, the SIE programme, the role of ML, the paths to simulation and publication |
| `02_STATUS.md` | what is true now: what runs, the numerical standard (Part A §2), gates, known-open rows, open items (Part A §7), ledger, improvements |
| `03_METHOD.md` | how it works: architecture, contract, numerics, precision, corpus schema, testing, cusp factorisation, the gates table, validation ladder |
| `04_ROADMAP.md` | what comes next: the roadmap by theme |
| `05_DECISION_LOG.md` | every decision `D-nn`: decision, reason, status, amended-by |

| Root file | What it is |
|---|---|
| `README.md`, `LICENSE`, `CITATION.cff`, `CHANGELOG.md` | entry point, the MIT licence, citation metadata, release history |
| `contract.py` | the frozen API — record, gate, diagnostic and scan types |
| `config.py` | *how hard to try*: `NumericsConfig` loaders, presets, the **setups table** — one `Setup` per scenario, levels `draft`/`standard`/`fine`/`reference` (D-73) |
| `physics_config.py` | *what to solve*: the scenario registry; never a numerical parameter |
| `run.py` | solves, gates, writes the corpus, exits non-zero on failure (G5.6); `--setups` lists them, `run.py he_atom_lda --res fine --device cuda` runs one |
| `figures.py` | publication figures from records; nothing drawn without a verdict |
| `test_suite.py` | the profiled suite (`quick`/`full`/`gpu`) over scenarios, gates and goldens |
| `scripts/` | benchmarks, scans and generators (`benchmark.py`, `figures_suite.py`, `scan_throughput.py`); hardware and numerics probes in `scripts/probes/` |

### 3. Five things a contributor needs to know

1. **CPU float64 is the audit reference.** Goldens, fingerprints and thresholds are defined there, so
   a CPU/GPU disagreement is a device-side defect until proven otherwise (`02_STATUS.md` Part A §2).
2. **Never raise a threshold to pass.** A threshold is a claim about the physics, not a dial.
3. **Derive the grid, do not configure it** (D-42, D-53, D-64): a single centre is box-limited, a
   molecule lattice-locked with `h = R/m`. `standard` *is* that production rule and the only level
   comparable with goldens and the known-open registry; `draft`, `fine` and `reference` are
   **exploratory**, routed through `cdft.grid.grid_rule(...)`, recorded in
   `grid_overrides["grid_rule"]`, banner-announced.
4. **Known failures are registered, not hidden.** A measured failure with a named cause keeps its
   verdict and loses only the power to fail the build (D-43); it is pinned in `cdft.gates.known_open`
   and `tests/golden/values.json` and fails again once it drifts out of band.
5. **Nothing emits a number without a verdict.** Gates run at solve time and travel with the data; a
   record that failed one is `INVALID` and excluded by the reader, and figures badge it.

### 4. Working rules

- Code, not physics lectures, unless asked; every non-obvious formula cites its survey tag.
- **Documents are state, not story** (D-73): a solved problem is deleted and recorded only as solved,
  in `05_DECISION_LOG.md`.
- **One home per fact** (D-73): open items `02_STATUS.md` Part A §7 · the numerical standard
  `02_STATUS.md` Part A §2 · decisions `05_DECISION_LOG.md` · what comes next `04_ROADMAP.md`.
- **Code comments are one-liners with `D-nn` pointers** (D-73); rationale only in `03_METHOD.md`
  Part E and the decision log.
- **Never raise a threshold to pass**, and **derive the grid** rather than configure it (§3).
- Optimise for speed, never at the cost of an auditable gate; an ungateable trick does not ship.
- Do not label the work "PINNs" externally (D-18, Part D §2).

---

## Part B — Baseline goals

The non-neural Kohn–Sham reference implementation: the instrument producing the training corpus and
the ground truth for a later physics-constrained / learned-functional project ([A17]; 2026-09-12).

### 1. One-sentence mandate

Build a real-space, GPU-resident, fully differentiable Kohn–Sham DFT solver for isolated molecules
producing **gate-certified** converged densities, potentials and energies fast enough to fill a
training corpus overnight on one consumer GPU. An unverified number is worse than a missing one.

### 2. What this project is, against the four alternatives

| Candidate | Verdict | Reason |
|---|---|---|
| **ground-state Kohn–Sham DFT** | **selected** | the only rung on which learned functionals reached real chemistry: DM21 [C1] and Skala [C2] replace E_xc in a KS-SCF loop — the "fully machine-learned" class of [A17], which places them on Jacob's ladder's 3rd–4th rungs ("rung 6" was this project's label, withdrawn) |
| real-time TDDFT | architected for, not built | a superset: propagators [H2], [H3] need only H·Ψ, so `Hamiltonian.apply` is the seam |
| orbital-free DFT | rejected | where ML is most needed (T_s[n]) but least accurate; feature pipeline kept compatible [C16]–[C18] |
| classical fluid DFT (Evans) | not this project | recorded because the folder name misleads |

### 3. Scope

#### 3.1 In scope for v1

Isolated 3D molecules, non-periodic, closed-shell, on a uniform Cartesian grid with a high-order
finite-difference Laplacian [D1], [D2] and a Dirichlet boundary on a masked domain; ONCV
pseudopotentials [E1]–[E4]; LDA [A7]–[A9], GGA [A4]–[A6] and meta-GGA [A11] native in torch with libxc
[I1] as oracle; open-boundary electrostatics [F15], [D14]; CheFSI [F1]–[F3] with LOBPCG [F4]; Pulay /
Anderson mixing [F8]–[F12] with a SAP guess [F13]. Every field, energy term, eigenvalue, occupation,
force and SCF step goes to an HDF5 record with provenance and a gate report. One GPU (6 GiB) and CPU,
one code path, audited mixed precision. Detail: `03_METHOD.md` Part A.

#### 3.2 Explicitly deferred, with the increment that adds it

| Feature | Increment | Why deferred |
|---|---|---|
| spin polarisation (collinear LSDA) | I4 | not in the v1 floor; promoted by D-51 |
| fractional occupations / smearing | I4 | occupations are floats; Fermi search [F20] with spin |
| forces → geometry optimisation, MD | I7 → I9 | needed egg-box control [D17], closed by D-58 |
| dispersion corrections D3/D4 [A18], [A19] | I8 | additive, post-SCF, separable |
| hybrid functionals (exact exchange) | not scheduled | costly on a grid; [C2] matches it semi-locally |
| PAW [E6], ultrasoft [E7] | not scheduled | not justified at 50–100 atoms |
| periodic systems, k-points | not scheduled | out of scope (D-02); Part E |
| RT-TDDFT [H1]–[H3] | I11, optional | reachable, not a v1 goal |
| Kohn–Sham inversion [G1]–[G7] | I10 | only for *potential* targets; core by D-17 |

#### 3.3 Out of scope permanently

Any PINN or learned-functional implementation (Q2.5, explicit); §3.4 states the boundary precisely
enough to be enforced.

#### 3.4 The boundary — this repository builds an instrument, not a model

**No machine learning of any kind**: a combined solver-and-learner cannot be debugged, since a wrong
number could come from the physics or from the fit. **In scope:** producing a converged,
gate-certified KS solution and writing it down, and **evaluating** a functional somebody else
trained. **Out of scope, permanently:** any training loop, loss or optimiser over functional
parameters; any network architecture; any dataset assembled for fitting; any accuracy claim resting
on a learned component. **Exactly three things cross**: the HDF5 corpus (`CORPUS_LAYOUT`,
`SCAN_LAYOUT`); `XCFunctionalProtocol` — a pure `evaluate`, no `fit`, `train` or `parameters()`; and
`NonlocalFeatureProtocol`, which computes features and their adjoint. Gate **G5.7** enforces it.

The solver is nevertheless differentiable, in one sense of two (D-47). *Inside the functional*: a
learned E_xc has no closed-form derivative, so `v_xc` comes from `torch.autograd.grad`, one backward
pass per SCF iteration, and `XCOutput` requires `v_xc`, `v_sigma`, `v_tau`, so the solver never
computes derivatives (**G0.5**). *Through the SCF* is needed only to train and is deferred: unrolling
measures 7.9 GB for water, 559 GB for 100 atoms, against a 6 GiB card ([C3], [C4]).

#### 3.5 What "exact" means here, and what it does not

**This solver does not produce exact solutions of the many-electron problem**, but converged solutions
of an *approximate* theory with certified numerics: 1e-8 Ha numerics says nothing about the functional
being right. Three things are genuinely exact; everything else is approximate but certified, and where a
scan's endpoints must beat DFT the labels are imported or computed with a correlated method (Part E §3).

| Genuinely exact | Why | Where |
|---|---|---|
| **analytic limits** — oscillator eigenvalues, Gaussian-charge Hartree energy, the uniform-gas exchange constant | closed form | G0.2, G1.1–G1.3 |
| **one-electron systems** — H, He⁺, H₂⁺ | the exact functional cancels Hartree self-repulsion exactly | D1.1 |
| **fractional-occupation conditions** — linearity in N [A21], flatness in M [A23] | theorems, built from the integer endpoints | D1.2–D1.4 |

### 4. Success criteria

Each maps to gates in `03_METHOD.md` Part C.

| id | criterion |
|---|---|
| **S1** | all Tier 0–3 gates pass on every registered scenario, on the CPU-float64 and GPU-mixed-precision paths, agreeing to tolerance |
| **S2** | the ladder of §5 within tolerance, with one cross-code comparison against Octopus [D5] on a dissociation curve, ε ≲ 0.06 ([B5]) |
| **S3** | a 12-hour unattended run on the GTX 1660 Ti gives ≥ 500 gate-passing, fully-recorded solutions of molecules ≤ 20 atoms, or ≥ 5 of 100 atoms; provisional until measured |
| **S4** | re-running an archived record from its config reproduces the total energy to < 1e-9 Ha on the same hardware, < 1e-6 Ha across devices |
| **S5** | no silent failure: every record carries a gate report, and one that failed a gate is `INVALID` and excluded by the reader |
| **S6** | publishable: docstrings on every public function, a citation on every non-obvious formula, an environment lock, a documentation trail in `docs/` |

### 5. The benchmark ladder

Each rung is a gate in `03_METHOD.md` Part C.

| # | System | Property | Reference | Requires |
|---|---|---|---|---|
| L0 | 3D harmonic oscillator | (n+3/2)ħω | analytic | Laplacian, eigensolver |
| L1 | particle in a box | π²(n²+m²+l²)/2L² | analytic | boundary conditions |
| L2 | Gaussian charge density | erf(r/√2σ)/r | analytic | Poisson solver |
| L3 | uniform electron gas | e_x = −(3/4)(3/π)^{1/3} n^{1/3} | analytic [A8] | XC layer |
| L4 | atoms (He, Be, Ne) | valence eigenvalues | ONCV files [E3] | pseudopotentials |
| L5 | H₂, LiH, H₂O, CH₄, N₂, CO | r_e, ω_e | published PBE | forces |
| L6 | same set | binding energy | Octopus/GPAW | full stack |
| L7 | H₂, H₂⁺ dissociation | shape, ε vs Octopus | [B5], [C1] | full stack |
| L8 | atomization energies, closed shell | vs W4-11 [B2] | after spin | I4 |

**L7 is the scientifically important rung.** Stretched H₂⁺ is a one-electron system LDA and PBE get
*wrong* by construction [A9]; even Skala's largest error is SIE4x4, 13.6 kcal/mol [A17]. Reproducing
that failure quantitatively validates as strongly as a success and is harder to fake.

### 6. Spin

Spin polarisation is **Increment 4**, directly after the SCF (D-51): atomization energies (L8, W4-11
[B2]) need open-shell atoms, and fractional spin is half the flat-plane condition (Part C). It is
cheap — the spin index exists from Increment 1 with extent 1 and the polarised forms are written, so
I4 adds the occupation machinery [F20], not a reshape.

### 7. Non-goals, stated to prevent scope drift

- Not a general-purpose electronic-structure package: molecules, non-periodic, semi-local.
- Not competitive in speed with VASP / Quantum ESPRESSO / GPAW: *fast enough to fill a corpus
  overnight* and *correct enough to trust*, in that order of difficulty and the reverse of priority.
- Not an accuracy record attempt: grid and pseudopotential errors put this at grid-code accuracy, so
  absolute totals compare with Gaussian-basis literature only as differences [B4], [B5].
- Not a place for undocumented cleverness: every function documented, no silent failure.

### 8. Hardware

The **CUDA development host** is a laptop GTX 1660 Ti (TU116, CC 7.5) under Windows, torch
2.11.0+cu128, MKL: 6 GiB GDDR6, ≈ 288 GB/s, ≈ 5.4 TFLOP/s fp32, **fp64 at 1/32 of fp32 ≈ 0.17
TFLOP/s**, fp16 at 2×; no tensor cores, so TF32 is unavailable and bf16 not native (CC ≥ 8.0), and
sustained clocks throttle by roughly 2×. The second host is a **2-core Linux CPU host** (8 GiB,
5.8 GiB cgroup limit, torch 2.14.0+cu130 on the CPU) with no GPU (D-74); GPU-bound work runs on the
CUDA host. Memory model for the orbital block, `n_states × n_grid × bytes`, three live in the
Chebyshev filter:

| System | h (Å) | domain | n_grid | n_states | fp32, 3 blocks | fp64, 3 blocks |
|---|---|---|---|---|---|---|
| H₂O | 0.15 | cube, 14 Å | 0.83 M | 5 | 0.05 GB | 0.09 GB |
| benzene | 0.18 | cube, 17 Å | 0.86 M | 18 | 0.17 GB | 0.34 GB |
| 100 atoms | 0.20 | cube, 27 Å | 2.46 M | 180 | **4.95 GB** | **9.90 GB** |
| 100 atoms | 0.25 | cube, 25 Å | 1.00 M | 180 | 2.01 GB | 4.02 GB |
| 100 atoms | 0.20 | **masked, 0.45×** | 0.88 M | 180 | **1.77 GB** | 3.54 GB |

Two conclusions, the least negotiable engineering decisions here. **A full-cube fp64 orbital block does
not fit** at 100 atoms — 9.9 GB against 6 GiB — so mixed precision decides whether it runs at all. **The
domain must be masked**: the union of atom-centred spheres (PARSEC [D3], Octopus [D5]) removes ~55 % of
points, free in accuracy since they are vacuum. Measured numbers are in `02_STATUS.md`; the model is
checked before a run, not after an OOM (R2).

### 9. Definition of done

> *"…overnight get accurate baseline for training PINNs on this problem with generally good quality
> of mesh/visualisation with publication ready plots and statistics."* — specification, Q11

Operationally: one unattended command solves the scenario list, writes the corpus with a gate report
per record, emits the convergence study and benchmark table, and exits non-zero on any failure.

---

## Part C — The self-interaction programme

All of it is **solver and data work**: no functional is trained here, and none of the five alterations
introduces one (Part B §3.4).

### 1. The hole

**Delocalization error**: E(N) should be *piecewise linear* between integers with a derivative
discontinuity at each [A21]; approximate functionals give a convex curve — barriers too low, gaps too
small, charge over-delocalised. **Static correlation error**: at fixed N the energy should be *constant*
between integer spin states; it curves, so dissociation and transition-metal spin states come out wrong.
The exact E(N, M) surface is a set of **flat planes** [A23], [A24], and the deviation from flatness is one
number for both errors. Skala's worst GMTKN55 subset is **SIE4x4 at 13.6 kcal/mol** against 3.89 [A17].

### 2. Why this hole is unusually worth attacking

**Weakest point of the state of the art**: beating Skala overall is a funded-laboratory problem, on its
own worst subset it is not. **Industrially expensive**: at 700 K a 13.6 kcal/mol barrier error is a rate
wrong by 1.8 × 10⁴ (Part D §4). **Exact data is free**: by [A21] the exact energy at fractional N is the
line between the integers and by [A23] at fractional spin it is constant, so **no correlated calculation
is needed at fractional occupation**, and for one- and two-electron systems the endpoints are analytic.

### 3. The theoretical constraint that drives the architecture

A semi-local functional evaluates E_xc from (n, ∇n, τ) pointwise and **cannot distinguish a one-electron
density from part of a many-electron one**, so it cannot know when to cancel the Hartree self-repulsion —
why SIE4x4 remains Skala's worst subset.

| Route past it | Extra ingredient | Cost on a uniform grid |
|---|---|---|
| **non-local density features** [C26], [C27] | integrals of n against smooth kernels | **O(N log N)** — FFT convolutions |
| projector / orbital-dependent corrections [C28], [C29] | occupation-space projectors | semi-local-ish, orbital-dependent |
| exact exchange (local hybrids, DM21 [C1]) | the exchange operator | O(N⁴) — ruinous here |

**The first route is the one to take, and the grid is why.** CIDER's features,
G_i(r₁) ∝ ∫ d³r₂ Φ(a(r₂), b_i(r₁), r₁₂) n(r₂) with Φ(a,b,r) = exp[−(a+b)r²], become **convolutions** when
the kernel is expanded in a few fixed exponents: spline-dominated and quadratic in an atom-centred code,
quasi-linear on a uniform grid [C27] — a few FFTs per SCF step here.

### 4. The five alterations

| # | Alteration | What it adds | Status |
|---|---|---|---|
| 1 | the XC layer accepts **non-local density features** | `NonlocalFeatureProtocol` computes features and owns their adjoint to a potential; `evaluate` gained `nldf`, `XCOutput` `v_nldf`, `XCSpec` `NLDFSpec`, whose `n_exponents` is the cost knob | v1.1 |
| 2 | **diagnostics are distinct from gates** | a gate asserts the *solver* is right, a diagnostic measures how a *functional* behaves — badly on purpose here. G4.5 requires stretched H₂⁺ to fail; the magnitude is the label a learned functional reduces, so a `Diagnostic` is **never thresholded** (D-20) | v1.1 |
| 3 | **scans are a first-class record type** | `ScanSpec` / `ScanArtifact` group runs into a curve, carry the `integer_endpoints` the exact reference is built from, and hold curve-level diagnostics | v1.1 |
| 4 | **promote spin polarisation** | fractional spin is the half of the flat plane that diagnoses static correlation error | D-51: I4 |
| 5 | **an all-electron / model-potential path** | the cleanest SIE systems are H, He⁺, H₂⁺, H₂, where a pseudopotential removes the core the error is defined against; point nuclei on the cusp-factorised path, plus soft-Coulomb variants | D-23, I1b |

### 5. The diagnostic suite

Parallel to the gates table of `03_METHOD.md` Part C, but nothing here has a threshold. D1.1 and D1.7
have grid-resolved forms and are worth far more per system than the scalars.

| ID | Diagnostic | Exact | Field? |
|---|---|---|---|
| **D1.1** | **one-electron self-interaction**: for one electron E_H[n] + E_xc[n] = 0 [A9] | 0 | **yes**, the integrand |
| **D1.2** | **fractional-charge deviation**: max deviation of E(N) from the line through the integers; the sign separates delocalization from localization [A21] | 0 | no |
| **D1.3** | **fractional-spin deviation**: max deviation of E(M) from constancy at fixed N [A23], [A25] | 0 | no |
| **D1.4** | **flat-plane deviation**: max ⎮E(N, M) − plane⎮ [A24], [C28] | 0 | no |
| **D1.5** | **derivative discontinuity**: (ε_LUMO − ε_HOMO) vs (I − A) from ΔSCF [A21] | 0 | no |
| **D1.6** | **Koopmans deviation**: −ε_HOMO vs the ΔSCF ionisation energy [A21] | 0 | no |
| **D1.7** | **asymptotic decay of v_xc**: fit −α/r^β in the tail, semi-local decaying exponentially [A16] | α = 1, β = 1 | **yes** |
| **D1.8** | **dissociation-limit error**: H₂⁺ → H + H⁺, H₂ → 2H, LiF → Li + F vs Li⁺ + F⁻ [C1], [A22] | analytic | no |
| **D1.9** | **barrier-height error** on a small near-exact set [B2], [K1] | ref. | no |

### 6. What to compute, and what it costs

Systems are small: H₂⁺ and H₂ in a 12 Å box at h = 0.20 Å sit on 60³ = 0.22 M points and need 2 and 5 MB
for three orbital blocks, a spin-polarised C atom (h = 0.18 Å, 67³, 6 states) 21 MB and LiF (14 Å, 78³,
8 states) 43 MB. The catalogue — one geometry per scan, the electron number swept:

| Scan | Points | Runs |
|---|---|---|
| fractional charge E(N) | 9 fractional N × 20 bond lengths | 180 |
| flat plane E(N, M) | 9 × 9 × 8 bond lengths | 648 |
| one-electron SI, H₂⁺ | 40 bond lengths × 9 fractional charges | 360 |
| | **total** | **1 188** |

At 20 s per run that is **6.6 GPU-hours — one overnight run**, in tens of megabytes of VRAM, with exact
references for every point and no correlated calculation anywhere. S3 was framed as *many molecules at
one functional*; the programme reframes it as **few systems, scanned densely over fractional occupation,
with exact references** (D-22).

### 8. Two cautions

**A functional that fixes SIE and breaks everything else is a failure**: regression-test any candidate on
ordinary thermochemistry first — GMTKN55 [B1] exists for this. **Do not extrapolate from two-electron
systems to chemistry**: H₂⁺ and H₂ are where the error is *cleanest*, not most *representative*, so the
catalogue must grow toward open-shell atoms and charge transfer.

---

## Part D — The role of ML and the PINN baseline

Work that happens **outside** this repository (Part B §3.4, G5.7); it is here so the instrument is built
with the right downstream use in mind.

### 1. The correct use of ML in DFT

**Point ML at ignorance, not at labour.** KS DFT is exact in principle [A2]: everything but E_xc[n] is
solved and costs time, not knowledge. Three roles, not interchangeable (D-16):

| Role | What it does | Beat what trained it? | Risk |
|---|---|---|---|
| **replace the unknown** — learn E_xc[n] or v_xc(r) | fills the hole | **yes** | high: it *defines* the answer |
| **speed up the known** — guess, mixing, preconditioner | same answer sooner | no, and must not | low: the solver checks it (G3.4) |
| **replace the solver** — geometry → energy | interpolates the solver | never | medium: silently out of distribution |

Only the first is science; most confusion in the field is the third called the first, and `contract.py`
keeps them apart as `InitialGuessProtocol` (path) and `XCFunctionalProtocol` (fixed point). Four rules
follow: a model's ceiling is whatever it was trained against, so labels come from above DFT or from
Part C §2; fields carry more information than numbers — [C19] trained neural LDA and GGA functionals on
inverted CI potentials from **five atoms and two molecules** and reached SCAN-level accuracy on hundreds
of unseen molecules; it must be clear whether the ML touches the answer or only the path; and functional
error must be separated from density error [B7], [B8], why `03_METHOD.md` Part A stores fields on the grid.

### 2. What "a baseline for training PINNs" means here, and what to call it

The solver produces what a baseline for training PINNs requires: gate-verified **labels and fields** — n,
∇n, τ, ∇²n, v_H, v_xc, v_ext, eigenvalues, energies with term decomposition and the exact references of
Part C — on which a physics-constrained network or learned functional is trained and judged. Training is
downstream; the solver is the referee. **Externally, do not title the work "PINNs for DFT" (D-18)**:
strict physics-informed networks have a credibility problem in scientific computing [J3], [J4] and fit
the KS setting poorly, putting ~150 mutually orthogonal 3-D fields against a Chebyshev filter [F1] doing
the same work in dense matrix products. What works is [C3], the KS equations as regularizer; the accurate
terms are *physics-constrained* or *constraint-satisfying learned functionals*.

### 3. Where a single researcher can still contribute

The scoreboard: Skala / Skala 1.1 [C2] reaches GMTKN55 WTMAD-2 ≈ 2.8–3.9 kcal/mol (W4-17 TAE MAE
≈ 1 kcal/mol) on molecules; CIDER26SS [C20] 4.10 across molecules, solids and surfaces; DM21 [C1] 3.97 on
molecules; the best conventional double hybrid (DH23) 1.7 [A17]. Beating these *overall* is a
funded-laboratory problem; beating one **where it is weak** is a single-student problem, and the weak
subsets are published — Skala's worst are SIE4x4 at 13.6 kcal/mol and DIE60 at 8.54 kcal/mol [A17]. Four
openings: **potential-target learning** ([C19] as template, needing a differentiable grid code and an
inversion module — why I10 is core); **basis-set contamination**, unmeasured, every ML functional having
absorbed its basis's artefacts; **discretisation consistency**, whether a functional trained at spacing h
is h-dependent; and **constraints by construction** — Lieb–Oxford [A10], scaling [A16] and linearity in N
enforced *architecturally*.

### 4. Industrial context

A barrier error is a *rate* error: 1 kcal/mol (chemical accuracy) is 5.4× at 298 K and 2.1× at 700 K,
5 kcal/mol (typical semi-local) 4 600× and 36×, Skala's 13.6 kcal/mol on SIE4x4 9 × 10⁹× and 1.8 × 10⁴×.
In the sector map [K1]–[K9] (`00_LITERATURE_SURVEY.md` section K) nearly every failure mode reduces to
**self-interaction / delocalization error** or **transition-metal spin states**, which is why G4.5
measures SIE. **The artefact with industrial value is the functional**, which travels between codes as
this solver never will: Skala reached CP2K through GauXC in about fourteen months [C21].

---

## Part E — From this corpus to real simulations

Isolated molecules calibrate an operator used in condensed phase because E_xc[n] is **universal**: PBE
came from exact constraints and the uniform gas, B3LYP from atomisation energies of small molecules, and
both are applied to proteins and solids. The corpus is a calibration, not a simulation: it holds no
interaction *between* molecules — electrostatics, Pauli repulsion, dispersion — which differs from
non-locality *inside* one calculation (Part C §3). The assumption is **transferability**, so coverage
matters more than volume (R8), and the SIE region is the choice.

### 3. The label problem — what this corpus supplies and what it does not

The solver produces the *exact Kohn–Sham solution for a given functional*, which is what the gates
measure — not a label for a *better* functional, since a converged PBE energy is a statement about PBE.

| What | Where it comes from |
|---|---|
| **ingredients** — n, ∇n, τ, ∇²n on a basis-free grid, and all potentials | this solver |
| **exact labels in the SIE region** — one- and two-electron systems, fractional charge and spin, the flat plane | this solver, free: exact XC cancels Hartree for one electron, and by [A21], [A23] fractional occupation lies between the integers |
| **exact labels for He, H₂** | `cdft.reference.literature`, PySCF totals in `reference.computed` |
| **energy labels for general molecules** | **elsewhere** — W4-11 [B2], GMTKN55, CCSD(T), NN wavefunctions [C24] |
| **v_xc(r) targets** | Kohn–Sham inversion (I10) |

Hundreds of molecules give hundreds of basis-set-free, gate-certified *densities and ingredients* — not
training targets, which are joined in from published benchmark sets as CIDER26SS was built. In the SIE
region none are needed.

### 4. The chain to a real simulation

**Yes — but through the functional, never through this solver.** (1) a solver reaching the exact KS answer
→ (2) self-consistency → (3) an XC layer that can carry a learned functional → (4) chemistry beyond H and
He, needing pseudopotentials (I5) or a short-ranged cusp factor → (5) open shells (I4) → (6) training
labels for general molecules (§3, from outside) → (7) the corpus at scale (I9) → (8) *the separate
training project* → (9) **forces** (I7) → (10) validation against real benchmarks (I8). Per-link state:
`02_STATUS.md`. **Link 9 is where real simulation begins**: dynamics is ground-state DFT plus forces
driving Born–Oppenheimer MD, so forces are the gateway to every use case in the sector map — which is why
egg-box control was closed (D-58) before I7. **The sector map is mostly periodic and D-02 excludes
solids**; a functional calibrated on molecules can be evaluated in a periodic code but not **validated**
here, so no materials claim rests on molecular validation alone.

### 5. Worked example: graphene

Not with this solver: without k-points the Dirac cone at K — the quantity of interest — does not exist
(D-02); Bloch states at general k are complex while the solver is float64 *real*; Dirichlet on a masked
domain is wrong for a crystal; a zero gap needs Fermi smearing (I4); and charge sloshing is the pathology
Kerker preconditioning [F11] addresses, off by default. Three routes work: **finite flakes** (PAHs are
molecules; needs I5 and I4, edge states dominating below ~135 atoms), **the functional travels** to a
periodic code, and **graphene-adjacent SIE targets** — defects, adatoms, adsorbate charge transfer,
nanoribbon gaps, edge magnetism [K6].

### 6. Verdict

For **molecular** simulation the path is standard and the links above are all of it; for **materials** it
runs through someone else's periodic code. For **the scientific result** — the flat plane on a basis-free
grid — the path is short, needs none of links 4, 7, 9 or 10 and, subject to kill-check 5, is unpublished.

---

## Part F — The path to publication

### 1. The asset

At fractional occupation the exact answer is known without any correlated calculation ([A21], [A23]),
while everyone else needs CCSD(T) labels and an industrial budget. Three assets compound it. **No basis
set**: no BSSE in any energy difference, and every quantity here is a difference. **The two-centre
oracle** (D-41): for H₂⁺ at *any* bond length prolate-spheroidal separation gives the exact wavefunction
to 2.2e-9 Ha, hence the exact density, hence the exact KS potential in **closed form**,
`v_s = ∇²√n/(2√n) + ε` with `v_xc = −v_H`. **Grid-resolved diagnostics**: D1.1 and D1.7 have *field*
forms, and a field says **where in space** the error is. Asset 2 is what to build the paper on.

### 2. Novelty assessment — what a referee will say

| Claim | Verdict | Evidence |
|---|---|---|
| the flat plane is violated by LDA/GGA | **long known** — Mori-Sánchez, Cohen, Yang ~2008–09; [A21]–[A25] | survey |
| flat plane recoverable at semi-local *cost* | known, [C28] (2017) | survey |
| flat plane computed **basis-set-free on a real-space grid** | **nothing surfaced** | search, 2026-09-13 |
| Kohn–Sham inversion is an active 2025–26 field | **yes, crowded** — density-matrix penalisation, guarantees, invDFT | search |
| real-space inversion is novel | **no — invDFT already does it** | Comput. Phys. Commun. (2026) |

The question to answer first: *the violation has been known for fifteen years; recomputing it with a
different discretisation — what is new?* **"Basis-set-free" is not sufficient**: for H₂ and H₂⁺ a large
Gaussian basis errs by ~1e-4 Ha against a self-interaction error of 1e-2 to 1e-1 Ha, orders of magnitude
*below* the effect. Basis-freedom bites in exactly two places and the paper must live in them: **diffuse
densities at fractional charge**, badly represented by Gaussian bases and natively handled by a large box,
and **energy differences along dissociation**, no BSSE.

### 3. The recommended paper — and it is not the flat plane

**Title, approximately:** *Where does delocalization error live? Spatially resolved exchange–correlation
potential errors at fractional charge, from exact references.* For H₂⁺ the exact v_xc is closed-form at
*any* bond length and fractional occupation; against LDA, PBE and r2SCAN **as a field** over the (R, N)
plane it shows not *how large* delocalization error is but *where in space it sits* as charge leaves a
stretching bond, the scalar violation being the validation. Five figures carry it: (1) E(N) for H₂⁺ at
several R against the exact straight line, validating the known convexity basis-free; (2) the flat plane
E(N, M) for H₂, validation plus the r2SCAN surface if it is unmapped; (3) **v_xc^exact − v_xc^approx along
the bond axis as a function of (R, N)** — the result; (4) the D1.1 self-interaction integrand as a field,
showing where the functional self-interacts; (5) the v_xc tail decay of D1.7, exact −1/r against the
semi-local exponential. **H₂⁺ alone is a complete paper with zero external data**; H₂ extends it and needs
one correlated density from outside, so it is the second half, not a dependency of the first.

### 4. The critical path

Molecules at chemical accuracy (done) → the XC layer, LDA and PBE (done; r2SCAN deferred) →
self-consistency (done) → fractional occupation (needs the scan driver) → spin (I4) → **exact v_xc for H₂⁺
in closed form (I10, about a day: `v_s = ∇²√n/(2√n) + ε`, no inversion machinery)** → scan machinery (I9)
→ the 1 188-run catalogue (6.6 GPU-hours, Part C §6). **Not needed for this paper:** I5, I7, I8, the
corpus at scale, GPU residency. Scheduling: `04_ROADMAP.md`.

### 5. Kill-checks — do these before building, in this order

- [ ] **1. Has the (R, N)-resolved exact v_xc map for H₂⁺ been published?** Search hardest against
      **Baerends, Gritsenko and Buijse**, who own the *structure* of the exact KS potential; if it exists
      the novelty collapses to the fractional-charge axis. **Highest risk.**
- [ ] **2. Is r2SCAN's flat plane already mapped?** If yes, figure 2 is validation only.
- [ ] **3. Does the diffuse-density argument hold quantitatively?** E(N = 0.9) for H₂⁺ in a large Gaussian
      basis against the grid; agreement to 1e-5 Ha means the argument must be dropped, not defended. Half
      a day, and it decides the framing.
- [ ] **4. Does invDFT already produce figure 3?** Read the paper, not the abstract.
- [ ] **5. MRChem / MADNESS** ([D22], Jensen et al., *JPCL* 8, 1449 (2017)) give basis-set-free
      all-electron energies to μHa: any fractional-occupation or flat-plane results there? They are also
      the best oracle for interacting H₂ (O-20).

**If check 1, 3 or 5 goes badly** the fallback is a **data paper**: the 1 188-run corpus with exact
references, gate-certified and basis-set-free, is publishable in its own right (*Scientific Data*, or the
JCTC/JCP dataset track) and is what the downstream project needs anyway.

### 6. Expectations

**A strong MSc thesis and a solid JCTC or JCP paper**, not a Nature or Science result. What is new is the
*reference quality*, the *spatial resolution* and the *two-dimensional coordinate*. **The larger prize is
not the paper**: the same 1 188 runs are the thesis result, the figures and the training set for the
functional project. The cautions of Part C §8 apply to the paper too.

---

## Part G — Open problems and method limitations from the literature

Positions from the survey; tags and the ML-XC literature-map insights resolve in
`00_LITERATURE_SURVEY.md` (section L for the map).

| # | Open problem | Position |
|---|---|---|
| P1 | one-electron SIE is not removable semi-locally; SIE4x4 is Skala's worst subset, 13.6 kcal/mol [A17], [C2] | **the target**: references are free at fractional N and M [A21], [A23] |
| P2 | non-local features are the affordable route past P1 — O(N log N) on a grid vs quadratic in Gaussian bases [C26], [C27] | **nobody has built them on a grid** |
| P3 | the label ceiling: a converged PBE energy trains PBE; better labels need CCSD(T)/FCI/NN wavefunctions [B7], [C24] | Part E §3; in the SIE region none are needed |
| P4 | potential-target learning needs v_xc(r) → inversion, ill-posed and crowded [C19], [G4]–[G6] | one electron sidesteps inversion (`v_xc = −v_H`), many-electron does not |
| P5 | basis-set contamination and discretisation consistency of learned functionals, unmeasured | grid codes only; cheap and never done (Part D §3) |
| P6 | real-space silent failures: egg-box [D17], cusps [D1], mixed precision on consumer GPUs [I6]–[I8] | egg-box closed (D-58); cusp solved at one and two centres (D-54) |
| P7 | differentiable SCF: unrolling is memory-infeasible, implicit differentiation is the route [C5] | descoped (D-47); the fixed-point map stays extractable |

Method limitations behind the choices here. **Gaussian-basis KS** (PySCF) carries BSSE into every
difference and needs aggressive diffuse sets at fractional charge: oracle only [I2], compared on
differences [B4]. **Real-space FD with pseudopotentials** has the egg-box error and removes the core the
SIE is defined against. **A bare −Z/r on a grid** converges as O(h¹) and was replaced by **cusp
factorisation** (D-35), itself limited by the multi-centre `W` jump, exponential growth of φ for orbitals
decaying slower than e^(−Zr), a residual 1/r from GGA potentials and extra Hellmann–Feynman force terms —
met by cell averaging (D-39), cusp-aware quadrature (D-54) and, past row 1, a short-ranged factor or
pseudopotentials (D-52). **Multiwavelets** (MRChem, MADNESS) are not a training substrate but are novelty
risk and oracle at once (Part F §5, check 5). **LOBPCG** stagnates because TPA describes −½∇², not the
transformed operator (fixed in the similarity frame, D-56). **KS inversion** is ill-posed and already
exists in real space, so one electron is closed-form and two use a PySCF density. **CIDER features** carry
a spacing-dependence risk (R4): physical units plus a two-spacing gate. **Differentiable DFT** needs
7.9 GB to unroll water on a 6 GiB card, so implicit differentiation and the extractable fixed-point map are
the route. Two items the survey originally missed are now carried there with their `[USE]` notes: the
transcorrelated lineage of the cusp factorisation, to be cited before claiming the technique, and the
nuclear divergence of GGA potentials ([A26]; measured here for PBE).

---

## Part H — Specification answers still in force

From the original questionnaire (2026-09-12); only answers that still bind and are not already in
Parts A–B.

| Q | Answer, as binding today |
|---|---|
| Q1.5 | the chemistry is picked in one place, `physics_config.py`; numerical parameters and initial conditions in `config.py`, per-scenario settings separated from shared ones |
| Q1.6 | validity is decided by physics gates set from the literature, not by inspection |
| Q2.5 | the numerical method is state of the art from the literature, **excluding PINNs** |
| Development model | AI-assisted implementation under the author's direction and review; every result is checked by the gate catalogue against analytic limits, independent oracles and published reference data; `contract.py` is hand-written scaffolding |
| Q3.4 | success: state-of-the-art systems in reasonable time on a single consumer device, maximal information saved per run, benchmarks repeatable |
| Q4.3, Q4.4 | the code must be readable enough to publish; code, not physics explanations |
| Q5.2 | notes in `docs/`; git is the history |
| Q5.4, Q6.1, Q6.2 | the test suite grows with the code, not specified up front; continuous iteration; sized for a small team of 2–3 |
| Q6.4 | code **plus tests plus validation**; correctness rests on the gates, not on a line-by-line read |
| Q7.2 | optimise for speed: this is a baseline that must produce training data |
| Q7.3 | a short docstring on every function; the explanation lives in `docs/` (D-73: one-liner comments plus `D-nn` pointers) |
| Q8.1 | the red flag: silent failure — unphysical results that look publishable |
| Q10.1 | a scaffolded plan adding one feature at a time with little supervision, producing code reusable later |
| Q11 | project *Classical DFT solver*; author Václav Svoboda, Mathematical Physics master's student, junior researcher in the N. J. Mauser group, WPI Vienna; runtime at most overnight on a single consumer GPU |
