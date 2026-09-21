# 03 — Method and code: architecture, the all-electron path, the gates, the code rationale

How the instrument works. Part A is the architecture: layout, the frozen contract, the numerical
choices, the precision policy, the corpus schema, the testing layers. Part B is the cusp-factorised
all-electron path. Part C is the physics-gates specification table, machine-checked against
`cdft.gates.catalogue` by `tests/test_gate_catalogue.py`, so its `| **Gx.y** |` rows are edited only
together with the catalogue. Part D is the validation ladder and Part E the code rationale, module by
module, which every module docstring points at. Elsewhere: `01_PROJECT.md` what and why;
`02_STATUS.md` what is true now (Part A §2 the numerical standard, Part A §7 the open items, Part B
the accuracy ledger); `04_ROADMAP.md` what next; `05_DECISION_LOG.md` the D-entries and their
measurements.

---

## Part A — Architecture, layout, contract, numerics, precision, corpus, testing, platforms

Section numbers are stable: code cites `(Part A) section 6`, `section 7` and `section 8`. §4 (the
increments) lives in `02_STATUS.md` Part F and in `04_ROADMAP.md`; §9 holds the platform notes.
Package `cdft`: real-space Kohn–Sham DFT for isolated molecules, PyTorch, single consumer GPU, built
in twelve increments, each closed by a named set of gates from Part C.

### 1. Architecture in one page

```
  physics_config.py (what to solve)   config.py (how hard to try)
                       │                      │
                       └──────────┬───────────┘
                                  ▼
   Structure ─► Grid ─► [Pseudopotentials] ─► Hamiltonian ─► Eigensolver
   (atoms)     (mask, h)  (ONCV, KB, I5)      │ apply(Ψ)     (CheFSI / LOBPCG)
                                              │                    │
                                     Poisson ─┴─ XC                │
                                    (v_H[n])   (v_xc, e_xc)        │
                                        └────► SCF loop ◄──────────┘
                                          (mixing, occupations)
                                                │
                        Energies ── Forces ── Gates
                                                ▼
                                    RunArtifact ─► HDF5 corpus
```

**The one seam that matters.** Everything above talks to the Hamiltonian through `apply(psi) ->
H·psi`; it is never materialised as a matrix (D-06), so memory scales as `n_states × n_grid` and a
real-time TDDFT propagator [H2], [H3] needs nothing below this line. The path inside the seam stays
in `torch`: a learned functional needs autograd *inside* it (D-47), and `SCFStepProtocol` must
stay a pure differentiable function of the density so a downstream project can differentiate the SCF
implicitly [C5] (D-49). Backpropagation *through* the SCF is out of scope, the unrolled route being
memory-infeasible [C3], [C4].

### 2. Repository layout

```
cdft/
├── contract.py        # the frozen API — §3
├── config.py          # NumericsConfig loading, the named presets, the setups table (D-57, D-73)
├── physics_config.py  # ScenarioRegistry: the 16 scenarios and what to measure (D-57)
├── run.py             # python run.py <scenario …> [--setups] [--res LEVEL]: solve, gate, write
├── figures.py         # python figures.py <scenario>: solve, gate, record, publication figures (D-72)
├── test_suite.py      # the quick / full / gpu profiles; exit code is the verdict
├── pyproject.toml, README.md, LICENSE, CITATION.cff, CHANGELOG.md
├── docs/              # 00_LITERATURE_SURVEY, 01_PROJECT, 02_STATUS, 03_METHOD (this file),
│                      # 04_ROADMAP, 05_DECISION_LOG
├── src/cdft/
│   ├── config.py, physics_config.py, run.py   # shims onto the root modules (D-57)
│   ├── constants.py, structure.py, precision.py, device_policy.py, grid.py
│   ├── operators/   # laplacian, divergence, cusp, gauss_legendre, quadrature, fused,
│   │                # geometry_cache, external, poisson, hamiltonian (the seam)
│   ├── xc/          # base, lda, gga, mgga (r2SCAN deferred, D-55), dispatch, oracle (libxc)
│   ├── eigen/       # measure, chefsi, lobpcg, rayleigh_ritz
│   ├── scf/         # noninteracting, step, loop, mixing, solve, occupations
│   ├── gates/       # base, catalogue, known_open, runner, functional, tier0…tier5
│   ├── reference/   # literature, computed (GENERATED), radial, radial_ks, two_centre
│   ├── io/          # hdf5 (the corpus), provenance (G5.1)
│   └── diagnostics/, inversion/, observables/, pseudo/, td/   # placeholders, end of Part E
├── tests/           # pytest; tests/golden/values.json is the pinned layer
├── reports/         # run artefacts; git-ignored except fingerprint_cpu_reference.json
└── scripts/         # below; the probes live in scripts/probes/
```

`scripts/`, one line each: `benchmark.py` kernel and scenario throughput per host and device;
`dissociation_error_scan.py` error against the two-centre oracle along the bond (A-2);
`figures_suite.py` `figures.py` for every scenario, one process each; `fingerprint.py` the
full-precision fingerprint of every scenario; `interacting_ladder.py` refinement ladders against
NIST; `preset_sweep.py` eigenvalue error against box and spacing (D-42's table); `profile_solve.py`
one solve under `torch.profiler`; `pyscf_reference.py` regenerates `src/cdft/reference/computed.py`
(O-12); `scan_throughput.py` runs per hour of the scan shape (D-71); `sync_census.py` host reads and
kernel launches without a GPU (what G5.9 budgets). In `scripts/probes/`: `blas_probe.py` which
BLAS/LAPACK entry points survive; `cusp_factorisation_probe.py` the bare −Z/r ladder (D-32);
`gemm_probe.py` the Rayleigh–Ritz pieces in isolation; `interacting_grid_probe.py` the measurement
behind D-53/D-54; `stretched_h2plus_mixing_probe.py` the mixing trials on `h2plus_R8_lda` (O-22);
`trsm_probe.py` the wide triangular solve (D-62).

**`config.py` vs `physics_config.py`** — required by Q1.5 and strictly enforced: `physics_config.py`
answers *what system and what physics*, `config.py` *how hard to try*. A scenario never contains a
grid spacing, a numerics config never an atom, and the corpus generator sweeps the cross-product.

### 3. The contract

`contract.py` at the repository root is the frozen API: dataclasses for configuration, `Protocol`
classes for every replaceable component, the `RunArtifact` and gate types, and the on-disk schema. It
lives outside `src/` because it is a specification the implementation is written against (D-15).
Current version **1.6.1** (D-80: `InversionProtocol` without the copied gate pair; D-60 added the
additive provenance device block); 1.5.0 (D-49) defines `ComponentGateProtocol` and `SCFStepProtocol`
— `step(density, functional) -> (n_out, EigenResult, EnergyBreakdown)` and `residual(...)`, pure,
free of mixer state, no in-place mutation. `cdft.CONTRACT_VERSION_EXPECTED = "1.6.1"`. An amendment
requires an entry in `05_DECISION_LOG.md` and a bump of `CONTRACT_VERSION`, which is written into
every HDF5 record; older records stay readable, the corpus being append-only.

### 5. Numerical method choices, with reasons

| Component | Choice | Alternative | Why | D |
|---|---|---|---|---|
| Discretisation | uniform grid, order-2p (default 8) finite differences [D1], [D2] | Gaussian or plane-wave basis | no basis set ⇒ no basis-set superposition error in the energy *differences* that are all the project compares [B4]; one convergence parameter; a hot path of strided sums | D-03 |
| Domain | union of spheres around atoms | the enclosing cube | ~55 % of the points removed at zero accuracy cost, as in PARSEC [D3] and Octopus [D5] | D-08 |
| Boundary | Dirichlet; odd reflection at a hard wall | assuming the halo | the halo *is* the boundary condition; the decay is gated (G3.2) | D-33 |
| Nuclei | bare −Z/r, all-electron, cusp-factorised | pseudopotentials in Phase 1 | it removes the core that self-interaction error is defined against and breaks commensurability with published all-electron totals | D-23, D-35 |
| Grid sizing | derived per scenario; every nucleus a lattice point | one fixed preset | atoms are box-limited and molecules spacing-limited, so one preset is wrong for one of them; an off-lattice nucleus changes the physics silently | D-42, D-53, D-64 |
| Hartree | Coulomb-cutoff kernel [F15]; the 1s part analytic | plain FFT; ISF [D14] | verifiable against an analytic Gaussian (G0.2) and free of periodic images | D-54 |
| Eigensolver | Chebyshev-filtered subspace iteration [F1]–[F3] | preconditioned Krylov | needs only H·Ψ; the work is dense GEMM on a block; nothing to tune | D-06 |
| Second solver | LOBPCG [F4] as a permanent oracle | a fallback used on failure | G2.5 needs 1e-8 Ha agreement, and a failure-only solver is never exercised where it matters | D-56 |
| Mixing | periodic Pulay [F9] | Anderson/DIIS [F8], [F10] | robust, cheap, SPARC's default [D9]; Kerker [F11] is implemented but **off** | — |
| Initial guess | superposition of atomic densities [F13] | — | cheapest reliable guess, and the baseline an SCF-accelerator variant must beat | — |
| XC on the hot path | native torch | libxc | a learned functional needs autograd of `E_xc`; libxc is C, CPU-only and opaque to it, and stays the 1e-10 oracle (G0.6) | D-05, D-47 |
| Functional set | PW92, PBE, r2SCAN (deferred) | empirical hybrids | one per rung of [A3], non-empirical, each with a published constraint list to gate against | D-55 |

### 6. Precision policy

The hardware forces this (`01_PROJECT.md` (Part B, Goals) §8): on the GTX 1660 Ti fp64 runs at 1/32
of fp32, and a full-cube fp64 orbital block for 100 atoms needs 9.9 GB against 6 GB of VRAM. Mixed
precision is a feasibility requirement, not an optimisation (D-04).

**float32 (GPU, hot path)** — Hamiltonian application, FFTs, XC evaluation, Chebyshev filtering,
density accumulation. **float64 (always)** — the Gram matrix, Cholesky orthonormalisation, the
Rayleigh–Ritz subspace eigenproblem, every energy integral (accumulated in float64 whatever the
operand dtype), the SCF convergence test and the mixing history — `n_states²` or scalar work, cheap
even at 1/32 throughput, and each a known precision trap: [I8] documents loss of block orthogonality
as the way mixed precision breaks eigensolvers, hence G0.3 at 1e-10 unconditionally and a
`PrecisionConfig` that refuses to relax these stages. **float16** is unused (no tensor cores,
marginal dynamic range); **bf16 and TF32** are unavailable below compute capability 8.0, TF32
asserted off everywhere.

**What the solver does today.** Device-resident and float64 on the card, reproducing the CPU to 1e-13
(D-60); the float32 hot path is specified, not wired, `hot_dtype` returning float64 on every device
until it lands, so the provenance says what the run did (G5.1). Strict deterministic kernels cost ~2×
and run only inside G5.2's pair (D-61, D-69); on CUDA the arithmetic may be re-associated (D-65) or
replayed from a graph (D-67), bounded by the unchanged invariant, GPU-vs-CPU float64 ≤ 1e-10. The
measured tables (`02_STATUS.md` Part A §2 and Part F) replace this specification wherever the two
disagree.

**The audit (G2.10).** Every scenario re-runs in full float64 on CPU at the same grid and **a breach
fails the run**. That is what makes mixed precision safe rather than merely fast, and the direct
answer to Q8.1.

### 7. The corpus schema

Designed backwards from the downstream target — *learn the XC functional* — and from [B7]: a corpus
that stores only energies cannot separate functional error from density error. One HDF5 file per
batch, one group per run; the authoritative layout is `contract.CORPUS_LAYOUT`. Per run: provenance
attributes (contract version,
status VALID/INVALID/MARGINAL, git sha, config hash, package versions, device, precision policy, wall
time, scenario id); `structure/`; `grid/` (origin, spacing, shape, mask, weights); `density/` (n,
grad_n, tau, optional lapl_n); `potentials/` (v_ext, v_H, v_xc, v_nl_diag); `orbitals/` (eigenvalues,
occupations, optionally psi — off by default for size); `energies/` (total, T_s, E_ext, E_H, E_xc,
E_nl, E_ii, E_disp, E_HF_diag); `forces/`; `scf/` (the per-iteration trajectory and mixing state);
`gates/` (the GateResult table with measured values).

Three design notes: densities and potentials are stored **on the grid, not summarised** (D-14), as
[C7] learns density → CCSD(T) energy and [C13] learns `v_xc` directly; **τ(r) is always stored**,
even at LDA, as the rung-3 ingredient [A3] a Skala-like functional [C2] or an orbital-free kinetic
functional [C16] consumes; and **the full SCF trajectory is kept**, the training signal for the
SCF-accelerator variant, the evidence behind G2.3, and the only way to diagnose a
converged-but-wrong run afterwards.

### 8. Testing strategy

Three layers, never conflated: **unit tests** (`pytest`, seconds) for pure functions, parsers, shapes
and edge cases; **gates** (Part C, seconds to minutes) for physics verdicts, run at solve time and
attached to every record, Tiers 0–1 also in CI; **benchmarks** (hours) for the Tier 3 convergence
studies and Tier 4 literature reproduction, archived per increment, never in CI. On Q5.4: **gates
from day one, unit tests as they become useful.**

Markers are declared in `tests/conftest.py`, so an unknown marker is a typo rather than a new
category. `fast` is the inner loop — 511 passed and 1 skipped (CUDA-only) in ~35 s on two cores —
and `slow` (three-dimensional solves at production settings) runs only with `--runslow` or through
`test_suite.py`, whose profiles are `quick` (CPU, the inner loop), `full` (every gate and pinned
value) and `gpu` (`full` on CUDA, error if there is none). `oracle` (needs PySCF/libxc) is
**deselected**, not skipped, unless `CDFT_ORACLES=1` (O-24, D-76), so the 30 libxc comparisons are
off every summary by default. **A profile decides which checks run; it never changes the settings
they run at** — an earlier version made the fast profile faster by coarsening the grid, which is how
a gate stops measuring the thing it is named after.

`pytest --rebless` rewrites `tests/golden/values.json` (33 pinned values, last re-blessed 2026-09-19
on the Windows/CUDA host) and prints a diff. Slow tests are never skipped during a re-bless, which
would leave stale pins behind, and a moved golden value is a physics change, re-blessed only with a
reason recorded in `05_DECISION_LOG.md`. On a host of two cores under a 5.8 GiB cgroup limit the slow
tier runs **one file at a time** and never beside another solve: a diatomic SCF or the G3.1 refining
ladder on the 59³ grid exhausts that memory and is OOM-killed.

### 9. Platforms and environment

* **The Windows/CUDA development host** (GTX 1660 Ti, 6 GiB, torch 2.11.0+cu128, MKL). PySCF has no
  wheels there: install `[dev]` and run the oracle legs under WSL2 with `--oracles` /
  `CDFT_ORACLES=1`; by default the libxc tests are deselected and G0.6 is off the plan and the record
  (O-24, D-76), and the stored references in `src/cdft/reference/computed.py` mean the production
  path needs none. **One BLAS stack per environment** is mandatory: a conda PySCF puts conda's MKL
  numpy beside the pip torch, two OpenMP runtimes load, and every numpy or scipy BLAS call after
  `import torch` aborts the interpreter (OMP Error #15). Fix: a pip-only environment
  (`04_ROADMAP.md`; B4); `scripts/probes/blas_probe.py` names the surviving entry points, and
  `KMP_DUPLICATE_LIB_OK` is not acceptable for an audit reference. The production path is LAPACK-free
  on the numpy side (`src/cdft/operators/gauss_legendre.py`, bit-identical to `leggauss` for orders
  1–240); 36 numpy-LAPACK tests of `test_gauss_legendre` skip on MKL. `expandable_segments` is
  unsupported there, so D-69's allocator half is a no-op (it matters for the O-22 peak), and MKL is
  not bitwise reproducible between calls on differently aligned buffers (defect T-4), so CPU-side
  bitwise assertions are structural.
* **Linux CPU hosts.** Two cores, 8 GiB under a 5.8 GiB cgroup limit, torch 2.14.0+cu130 on the CPU,
  no card: a diatomic SCF or a G3.1 ladder can exhaust that, which is why the slow tier runs one file
  per process (§8). The CPU reference fingerprint (`reports/fingerprint_cpu_reference.json`, 15
  scenarios, float64, two threads, measured 2026-09-18 at the D-75 tree) is taken on such a host.

---

## Part B — Cusp factorisation: the all-electron path

**Status: implemented and on by default** (D-35, contract 1.3.0; in staggered divergence form since
D-38). Not a mode and not a special case: with no nuclei it reduces bit-for-bit to the untransformed
operator, and a unit test asserts that. Headline, same grid and same solver for both paths:

| system | h | bare −Z/r | cusp-factored | factor |
|---|---|---|---|---|
| H | 0.40 | 1.05e-2 | **2.35e-6** | 4 500 × |
| H | 0.30 | 7.16e-3 | **2.95e-6** | 2 400 × |
| He⁺ | 0.40 | 7.77e-2 | **2.52e-9** | 3 × 10⁷ × |
| He⁺ | 0.30 | 6.47e-2 | **3.27e-9** | 2 × 10⁷ × |

The bare path converges as O(h¹·⁰⁵) for H and O(h⁰·⁶²) for He⁺
(`scripts/probes/cusp_factorisation_probe.py`; its former gate G1.10 is retired, D-50). The
transformed path is not limited by the spacing at all.

### 1. Why the cusp is there, and the transformation

Kato's condition: where the potential has a −Z/r pole the exact wavefunction must have a matching
kink, dψ/dr → −Zψ(0), equivalently d(ln n)/dr → −2Z. It is exact, functional-independent and holds
for Kohn–Sham orbitals too (G1.11). A stencil fits a polynomial through neighbouring points and no
polynomial has a kink, so its nominal order is irrelevant at a nucleus.

Solve for a smooth function instead: ψ = f·φ with f = exp(−u), u = Σ_a Z_a|r − R_a|, transformed by
similarity, A = f⁻¹Hf. With ∇u = Σ_a Z_a ŝ_a and ∇²u = Σ_a 2Z_a/s_a, `∇²f/f = |∇u|² − ∇²u` and
`−½(∇²f/f) + V = −½|∇u|²`: every pole the kinetic operator produces from the factor is annihilated by
the corresponding term of the external potential, for **any** number of nuclei and any charges, so

    A φ = −½∇²φ + (∇u)·∇φ − ½|∇u|²·φ

has no singular term. For a single nucleus |∇u|² = Z² identically, the transformed potential is the
constant −Z²/2 and the exact one-electron solution is φ = const. Three structural properties: A and H
have identical spectra (similarity transform — algebra, not a numerical hope); A is self-adjoint in
⟨a,b⟩_w = ∫f²ab; and n = Σf_i f²|φ_i|², so everything downstream is unchanged.

### 2. Why the plain factor and not a short-ranged one

A switched factor u = Zr/(1+br) also satisfies u′(0) = Z and cancels the pole while keeping the
weight from spanning many orders. Measured at h = 0.20 against the exact φ it is **three to six
orders worse**: 1.49e-5 (Z = 1, plain) against −8.20e-4 at b = 1.0, and 1.16e-8 (Z = 2, plain)
against −1.38e-2. Its transformed potential expands as −2Zb + 3Zb²·r near the origin, and r means
|r|: a term linear in |r| is a **cone**, so the pole is removed and a weaker kink is put back one
derivative up. With b = 0 the potential is exactly constant for one centre.

**The general condition (D-52 item 4).** With u = Z s g(s²) the cone coefficient of the transformed
potential is **6Z g′(0)**, so Zs/(1+bs) fails while any switch with g′(0) = 0 — e.g. g = exp(−s⁴/a⁴)
— is cone-free, its first non-analytic term being |r|³, milder than the multi-centre direction jump
already tolerated at O(h³·⁴). To be measured: sweep a on H and He⁺ against −Z²/2, then a bare Li
1s²2s non-interacting test. That experiment decides whether all-electron breadth is a short-ranged
factor or pseudopotentials (§4).

### 3. What limits it instead: the box, not the spacing

φ tends to a constant, so a zero-filled halo imposes the wrong condition on it and the residue is of
order **e^(−2ZR)/h²** for a box of half-width R — measured for hydrogen, 1.3e-7 at R = 10 and 1.3e-5
at R = 8. Accuracy is bought with box size rather than resolution, and refining the spacing *hurts*:
no truncation error is left to remove while round-off accumulates. Hence the sizing rule
(D-42): a single centre gets half-box `16/Z` and `h = 1/Z`, a molecule is the opposite. The atomic
rule is exact for the transformed *eigenproblem* only; anything that integrates the density needs the
interacting rule (D-53) and the cusp-aware quadrature (D-54).

### 4. Multi-centre: general, and weaker

The cancellation is exact for any geometry, but ∇u contains a direction discontinuity at each nucleus,
so for **several** nuclei |∇u|² has a bounded jump there — far milder than a pole (a jump at isolated
points costs O(h³) in an integral against O(h) for a cusp), but not nothing. Measured, H₂⁺ at R = 2
bohr, h = 0.25: **+3.4e-4 Ha** against +2.0e-5 for the hydrogen atom on the same grid. Two limits:

* **The plain factor cannot leave the first row** (D-52 item 4). φ = ψ/f grows as e^{(Z−κ)r} for any
  orbital with decay constant κ = √(−2ε) < Z: hydrogenic 2p at Z = 10 gives **1.9e9 at 4 bohr** and
  Li's 2s (κ ≈ 0.45, Z = 3) reaches e^{25} at 10 bohr, while He (κ ≈ 1.07) is fine. This is a
  Z-limit, not a "p orbitals arrive later" note: all-electron breadth is either the short-ranged
  factor of §2 or pseudopotentials (I5), and the §2 experiment on Li decides which.
* **Forces.** f moves with the nuclei, so the Hellmann–Feynman expression acquires extra terms (gate
  G6.1) that Increment 7 must derive — which is why the transform landed before forces, not after.

### 5. Side effects

* **The off-grid-nucleus pathology disappears for one nucleus and not for two.** For a single centre
  nothing depends on where the nucleus lands; for two, `W` carries the cross terms `2 Z_a Z_b ŝ_a·ŝ_b`
  and a unit vector is discontinuous *at* its nucleus — measured for H₂⁺ at R = 2 before cell
  averaging, 2.5e-3 on grid points, 3.7e-3 at a half-cell offset, 1.1e-2 at a *finer* spacing, a
  reading of where the nuclei landed rather than a convergence sequence. `W` and `∇u` are therefore
  cell-averaged within two spacings of a nucleus (D-39); the residue is G2.7's business (A-1, A-3).
* **G1.11 verifies that the factor was applied, not how well the grid resolves it** — 2.1e-10 (H) and
  4.8e-12 (He⁺) at a single centre, SKIPPED in a molecule where the capped fit window leaves too few
  points (A-7). At GGA level it must not assert −2Z (D-52 item 3), PBE's `v_xc ≈ −0.017/r` shifting
  the cusp to `Z_eff ≠ Z`, so `v_xc` is cell-averaged near nuclei like `W`.
* **The measure abstraction is reusable**: `cdft.eigen.measure.Measure` gives both eigensolvers an
  arbitrary SPD inner product, and the uniform path dispatches separately and is unchanged.

### 6. What it does not do

It does not make the solver better than the functional: it removes a *discretisation* error that was
larger than the functional error the project exists to measure. Nor does it integrate the density it
produces — `∫n·v_ext` has an integrable 1/r integrand and the total energy never needs it (D-52 item
2), so `E_ext` is reported as the exact complement and the record says so (`external_method`), while
every density integral needs the cusp-aware quadrature of D-54, the plain `h³` rule overestimating
`∫f²` by 15.4 % at `h = 1/Z` (A-9), which G1.13 measures. **It is never applied
to `SOFT_COULOMB`** (D-2, closed): `W = −|∇u|²/2` cancels `V = −Σ Z_a/s_a` term by term, and a
softened potential would leave the residual `Σ Z_a (1/s_a − 1/√(s_a² + c²))`, singular at each
nucleus, on the untransformed side — so `_use_cusp_factorisation` selects the factor for
`NUCLEAR_COULOMB` only, and a soft-Coulomb scenario runs on the plain operator, where its potential
is smooth and needs no factor.

### 7. Two lessons from the two forms that failed

1. **Self-adjointness must be built in, not approximated.** The collocation form of A measured a
   relative asymmetry of 5.6e-3 (H) and 1.0e-1 (He⁺), no better than the bare operator; a Galerkin
   form with the *central* first-derivative stencil was exactly symmetric and carried a checkerboard
   null state reporting −0.499955 Ha against a true −0.5 — the right eigenvalue with a contaminated
   eigenvector. The staggered divergence form (D-38) is self-adjoint by construction at any order for
   any weight and cannot annihilate the checkerboard.
2. **A varying measure amplifies an error a uniform quadrature would hide.** Zeroing the undefined
   `∇u` *at* a nucleus loses the magnitude `Z_a` at the one point where f² = 1, and cost 4.2e-3 (H)
   and 1.03e-1 (He⁺) at h = 0.3. The same omission was made twice, in the operator and in the
   kinetic-energy assembly; one cell-averaging routine now feeds both.

---

## Part C — Physics gates: the specification table

**Purpose.** Q8.1 names the failure mode this project most needs to prevent: *"silent failures, the
results become unphysical however look publishable."* This Part defines every check the solver must
pass, its tolerance and the literature the tolerance comes from.

**Contract.** A gate is a function `(RunArtifact) -> GateResult` with a verdict, a measured value, a
threshold and a citation. Gates are **run at solve time** and written into every record; a record
whose gates did not all pass is `status = INVALID` and is excluded from the training corpus by
default. No code path in `cdft` emits a number without a verdict attached. 56 gates catalogued in 7
tiers (0–6): 46 implemented, 4 deferred (G2.6, G2.8, G6.1 → forces, I7; G2.10 → the float32 hot
path), 6 out of Phase 1; current verdicts and the known-open registry live in `02_STATUS.md`
Part A §3.

**Tolerance philosophy.** Every threshold is **EXACT** (analytically known, the threshold set by
floating-point arithmetic), **DERIVED** (following from a convergence parameter under the project's
control and tightening as it tightens) or **EMPIRICAL** (from published practice, citation
mandatory), and the kind is always stated. Units are Hartree atomic units unless stated;
1 Ha = 27.211386 eV, chemical accuracy = 1 kcal/mol = 1.5936e-3 Ha. Rows carry the date of any
restatement.

**Rules for writing a gate**, each paid for by a failure of this project:

1. Gate the thing that matters, not a proxy for it.
2. If a gate's verdict depends on its own input ladder, it is not a gate.
3. A gate that can only be satisfied by accident is worse than no gate.
4. Never let a fast profile change the physics settings.
5. State which of the three a missing gate is: deferred with an owning increment, out of scope for the phase,
   or unknown to the catalogue. The third is drift.
6. A criterion justified for one operator must be *gated on* that operator, not merely documented as
   being for it (D-44).
7. A gate that reads a key must be tested against a payload that has it.
8. State a threshold's relationship to the known error — G4.9 is 2.5e-3 because the measured error is
   1.0e-3 to 2.0e-3, and a gate set below a recorded defect fails on something already scheduled.
   Never raise a threshold to pass; register a measured, owned failure instead (D-43).

**Two mechanisms behind the verdicts.** A gate a scenario names and this build cannot evaluate is a
**DEFERRED** verdict with a named owning increment, not a FAIL (contract 1.4.0, D-37). "Does this report fail
the build" has **one implementation**, `known_open.partition_failures`, which `cdft.run` and
`test_suite.py` both delegate to (D-45).

---

### Tier 0 — Algebraic. No physics. Runs in seconds, on every commit.

| ID | Gate | Threshold | Kind | Source |
|---|---|---|---|---|
| **G0.1** | **Laplacian convergence order.** *(Two readings were clarified in Increment 1 and both are implemented as described in `02_STATUS.md` Part F, row Increment 1: σ is fixed at 4·h_coarsest rather than tracking h, and the error norm is taken over \|r\| ≤ 3σ. Over the whole box the zero-filled halo makes the error grow as 1/h² and the fitted order is −2.74 for a correct order-8 stencil.)* Apply the order-2p finite-difference Laplacian to a Gaussian of width σ = 4h and compare with the analytic result. Fit log‖err‖∞ vs log h over four refinements. | measured order within 5 % of nominal 2p; default 2p = 8 | DERIVED | [D2] |
| **G0.2** | **Poisson solver against an analytic charge.** *(Both stated thresholds are enforced jointly; see `02_STATUS.md` Part F, row Increment 1. At a zero-padding factor of 2.0 the energy is exact to 5e-16 Ha while the far-field potential is wrong by 1.1e-4 relative — an energy-only reading passes and corrupts diagnostic D1.7. The default pad factor is now 2.5.)* For ρ(r) = Q(2πσ²)^(−3/2)·exp(−r²/2σ²), the exact potential is V(r) = Q·erf(r/(√2σ))/r and the exact Hartree energy is **E_H = Q²/(2√π σ)** (both verified symbolically; ∇²V + 4πρ ≡ 0). | ‖V_num − V_exact‖∞ / ‖V_exact‖∞ < 1e-6 and \|E_H,num − E_H,exact\| < 1e-8 Ha at σ ≥ 4h | DERIVED | [F15], [D14], [F16] |
| **G0.3** | **Orthonormality of the orbital block.** ‖Ψ†Ψ − I‖∞ after the Rayleigh–Ritz step. | < 1e-12 (tightened from 1e-10 by D-34, measured 1.3e-15 … 1.8e-15; the Gram matrix and Rayleigh–Ritz are unconditionally float64) | EXACT | [I8]; D-34 |
| **G0.4** | **Hermiticity of H.** For 32 random vectors, \|⟨φ\|H\|ψ⟩ − ⟨ψ\|H\|φ⟩*\|. | < 1e-10 in float64; < 1e-5 in float32 | EXACT | — |
| **G0.5** | **v_xc is the functional derivative of E_xc.** Every native potential is `torch.autograd` of the energy density (`SemiLocalFunctional`), so the check is autograd against a central finite difference of `∫e_xc` on random `(n, σ)` ingredients, unpolarised and polarised, for every functional: relative step `1e-4·\|x\|` per ingredient, channels contributing below `1e-8` of the total floored, channel ratios capped at 100, PZ81 excluded within `\|r_s − 1\| < 0.01` (its branches meet with a published 3.2e-5 discontinuity). Restated 2026-09-14, D-55 item 3. | relative error < 1e-4 (measured 5.8e-7, LYP cancellation noise) | EXACT | [A3]; D-55 |
| **G0.6** | **libxc agreement.** The native torch LDA/GGA energy densities and potentials (`e_xc`, `v_rho`, `v_sigma`) against libxc through PySCF on 10⁵ random points, unpolarised and polarised, at reduced gradients `s ≤ 3` and `s ≤ 8`, for every native functional and component (Slater, PW92 original/modified, VWN5, PZ81, PBE x/c, B88, LYP). Metric `\|Δ\| / max(1, \|ref\|)`. r2SCAN is DEFERRED (D-55 item 5) and not in the set. Off the plan and the record unless `CDFT_ORACLES=1` / `--oracles` (`GateSpec.requires_oracle`, O-24, D-76); requested without PySCF it reports SKIPPED with that reason. | scaled error < 1e-10 pointwise (measured 1.4e-11) | EXACT | [I1]; D-55 |
| **G0.7** | **Kleinman–Bylander projector normalisation and ghost-state screening.** Project the reference atomic pseudo-wavefunctions shipped with the ONCV file back through the separable operator. | reproduces the file's reference valence eigenvalues; no spurious state below the lowest valence level | EMPIRICAL | [E1], [E3], [E4] |
| **G0.8** | **No NaN, no Inf.** Every kernel output is checked once per SCF iteration in debug mode, once per run in production. | zero occurrences | EXACT | — |
| **G0.9** | **Gradient convergence order.** As G0.1, for the first-derivative stencil, which the divergence form of D-38 leans on at least as hard as on the Laplacian. | measured order within 5 % of nominal 2p | DERIVED | [D2] |
| **G0.12** | **Weighted self-adjointness of the operator.** `<a\|A\|b>_w = <b\|A\|a>_w` in the measure `<a,b>_w = int f^2 a b`, measured on the operator that generates the search space rather than on the subspace matrix it produces. Added with D-35; **supersedes G0.4 on the cusp-factorised path**, where the plain Hermiticity of a collocation operator is not the property that matters. | relative asymmetry < 1e-12 | EXACT | D-35; D-38 |

---

### Tier 1 — Exact physical limits. The system has a known closed-form answer.

| ID | Gate | Threshold | Kind | Source |
|---|---|---|---|---|
| **G1.1** | **Uniform electron gas.** Constant density n: every exchange functional must return −C_x·n^(4/3) with **C_x = (3/4)(3/π)^(1/3) = 0.7385587663820224**, PBE correlation must reduce to PW92 (modified) at σ = 0, and the three LDA correlations are compared with the Ceperley–Alder values PW92 tabulates at r_s = 2 and 10 (a separate 1e-3 tolerance — the fits are only good to ~1 %, and this leg checks the physics rather than the code against itself). | relative error < 1e-12 (measured 3.3e-16) | EXACT | [A8], [A10] |
| **G1.2** | **3D isotropic harmonic oscillator.** With V_ext = ½ω²r², Hartree and XC switched off, eigenvalues must be (n + 3/2)ω with degeneracy (n+1)(n+2)/2. | \|Δε\| < 1e-6 Ha for the lowest 10 levels; degeneracy splitting < 1e-7 Ha | DERIVED | analytic |
| **G1.3** | **Particle in a cubic box.** Eigenvalues π²(n_x²+n_y²+n_z²)/(2L²). | \|Δε\|/ε < 1e-5 | DERIVED | analytic |
| **G1.4** | **Grid isotropy.** The 3-fold degeneracy of a p-like level on a Cartesian grid. A Cartesian stencil breaks rotational symmetry; the splitting measures how badly. | splitting < 1e-5 Ha at production h; must *decrease* monotonically under refinement | DERIVED | [D2], [D6] |
| **G1.5** | **Charge normalisation.** \|∫n(r)dr − N_e\|, in the measure the orbitals are normalised in. On the cusp-factorised path that is the cusp-aware mass weights `ω` of D-54, which integrate the cusp weight exactly (G1.13), so since 2026-09-14 this holds by construction *and* means what it says; before D-54 it held by construction on a rule that was 15 % wrong (A-9). `charge_error_plain` records the plain-rule value beside it. | < 1e-12 electrons (tightened from 1e-8 by D-34; measured ≤ 3e-15) | EXACT | D-34, D-54 |
| **G1.6** | **Non-interacting limit.** With E_H and E_xc disabled, the SCF must converge in one iteration to the eigenvalues of the bare external potential. | identical to the direct diagonalisation, < 1e-10 Ha | EXACT | [A2] |
| **G1.7** | **Virial theorem** (all-electron path). Non-interacting: −V/T = 2. Interacting: `2T + V_ext + E_H + E_xc + ∫n r·∇v_xc = 0` in the form `2T + V = −(virial XC scaling term)`, the scaling term evaluated by the coordinate-scaling derivative of `E_xc` (`virial_xc_scaling_term`, LDA `∫(e_xc − n v_xc)`); at GGA level the σ term is not yet carried and the gate reports SKIPPED with that reason (I4 or the next quadrature work). | \|−V/T − 2\| < 1e-3 (non-interacting: measured ≤ 1e-9; interacting He LDA at h = 0.25: 3.1e-4, He⁺ 2.7e-4, H 3.6e-6) | EMPIRICAL | standard; D-54 |
| | *Note:* this gate is **disabled on the pseudopotential path**, where the non-local operator makes the plain virial relation invalid. It is listed to prevent a future contributor from "fixing" a violation that is not one. | | | |
| **G1.8** | **Lieb–Oxford bound.** `E_xc[n] ≥ −2.273·E_x^LDA[n]` for the *integrated* energies of every converged density (`lieb_oxford_integral = E_xc / E_x^LDA` on the record). An artifact gate, not pointwise: B88 violates the local bound at large reduced gradient by construction (a published property), so a pointwise gate would fail every GGA run for a reason that is not a bug (D-55 item 4). | ratio ≤ 2.273; a violation marks the record INVALID | EXACT | [A10], [A16]; D-55 |
| **G1.9** | **Uniform coordinate scaling.** For n_λ(r) = λ³n(λr), exchange must scale exactly: E_x[n_λ] = λ·E_x[n]. The scaled density sampled at spacing `h/λ` is the same array times λ³ with σ times λ⁸, so the identity is tested exactly on a grid (the trick of G1.12) for Slater, PBE and B88 exchange at λ = 0.5, 2, 3. | relative error < 1e-12 | EXACT | [A10], [A16], [A17] |
| **G1.11** | **Kato cusp condition.** `ln n` fitted by a quadratic in r over a window around each nucleus (one to three spacings out, capped at a fifth of the nearest internuclear distance; linear where the window holds fewer than four distinct lattice radii, as on H₂⁺ at R = 2); the linear coefficient must be −2Z. Restated 2026-09-14 (D-55 item 1): the quadratic fit, because the self-consistent density has curvature the linear fit read as a 10 % slope error where the cusp is correct to 8e-3; **at GGA level the gate reports SKIPPED with the measured slope** — PBE's `v_xc` diverges as ≈ −0.017/r at a nucleus, so the exact GGA density does not obey −2Z (D-52 item 3). Asserted on the non-interacting and LDA paths. | relative deviation from −2Z < 0.1 | EMPIRICAL | Kato (1957); D-52, D-55 *Restated 2026-09-16 (D-64):* the density is read on the **recorded** grid (`gates.base.artifact_grid`), not a re-derived one, and shells are counted at `h/10` rather than 1e-6 bohr — an off-lattice nucleus splits every cubic shell and handed the quadratic a near-collinear design (the 229 shells measured on `h2plus_R2_lda`); `nucleus_offgrid_bohr` is in the detail. |
| | *Note:* the fit window is capped at a fifth of the nearest internuclear distance, because a window reaching towards a neighbour measures that neighbour's density — uncapped, H₂ at R = 1.4 reported a spurious 19 % violation; where that leaves fewer than eight points the gate reports SKIPPED. | | | |
| **G1.12** | **Coordinate scaling of the non-interacting kinetic energy.** T[ψ_λ] = λ²T[ψ]. Tested exactly rather than approximately: ψ sampled at spacing h and ψ_λ sampled at h/λ are the *same array*, so the relation carries no interpolation or truncation error at all and the threshold is round-off. | relative error < 1e-12 | EXACT | [A10]; Levy and Perdew, PRA 32, 2010 (1985) |
| **G1.13** | **Cusp-weight quadrature.** On the cusp-factorised path the density is `n = f²\|φ\|²` and φ is normalised in the *discrete* weighted measure, so G1.5 holds by construction whatever the rule. This gate compares the rule the solver normalises in — since D-54 the cusp-aware mass weights `ω` — against the closed form of `∫f²` — `π/Z³` for one centre, `(πR³/4)e^{−a}[2(1/a + 2/a² + 2/a³) − 2/(3a)]` with `a = 2ZR` for two equal centres — minus the numerically integrated tail outside the grid's box (D-55 item 6; 2.8e-8 relative for hydrogen on its 20.7-bohr box, 2e-11 for helium on 14). The plain `h³` value stays recorded beside it (`cusp_weight_integral_plain`, 15 % wrong at `h = 1/Z`). Other geometries report SKIPPED. Companion test: `E_H[1s] = 5Z/16` to 1e-8 through the same rule (`tests/test_quadrature.py`). | relative error < 1e-8 (atoms 1.2e-10; molecules **2.74e-9** (`h2plus_R2`, h = 0.25) and **2.93e-9** (`h2_R1.4`, h = 0.2333) since D-75, the mass weights' far rule lumped where spheres overlap; before it 2.290662e-08 and 2.032007e-08, known-open) | EXACT | closed-form integrals; A-9, D-50, D-54, D-58, D-75 |
| | *Note:* the threshold states what the cusp-aware quadrature (D-54) must deliver before any density integral — Hartree, XC, external — is trusted; it is not what the plain `h³` rule achieves at any affordable spacing (6.6e-4 at h = 0.25, 4e-5 at h = 0.125). Molecular rows are registered known-open (O-19); `02_STATUS.md` Part A §7 carries the current values. | | | |

---

### Tier 2 — Internal consistency of the converged solution.

| ID | Gate | Threshold | Kind | Source |
|---|---|---|---|---|
| **G2.1** | **Harris–Foulkes agreement.** The band-structure (Harris–Foulkes) energy and the self-consistent total energy must coincide at convergence. Their difference is second order in the density error, making it an unusually sensitive convergence probe. | \|E_tot − E_HF\| < 1e-6 Ha | DERIVED | [F22] |
| **G2.2** | **Janak's theorem.** ∂E/∂f_i = ε_i, checked by re-converging with the HOMO occupation shifted by ±0.01 (`fixed_occupations`) and central-differencing the total energy (two extra solves; `full` profile). | \|∂E/∂f_i − ε_i\| < 1e-4 Ha | EXACT | [F21] |
| **G2.3** | **Variational tail.** Over the final 5 SCF iterations the total energy must not rise by more than the SCF energy tolerance and the residual norm must decrease strictly. DIIS is permitted to be non-monotonic *before* that window. Reads the recorded per-iteration history. Kept strict when it failed on hydrogen at LDA: the cause was eigenvector noise from too few filter steps per SCF iteration, fixed in the solver (D-55 item 8). | zero violations in the tail | EMPIRICAL | [F8], [F12]; D-55 |
| **G2.4** | **Eigenvalue residual.** max_i ‖Hψ_i − ε_iψ_i‖₂. *(Where the eigenvalue comes from a **Galerkin** subspace matrix rather than from ``apply`` — the cusp-factorised path of D-35 — the residual measured with ``apply`` no longer bounds the eigenvalue error, so the gate thresholds the Ritz-value drift instead and records both. He⁺ is right to 8e-11 Ha with a collocation residual of 1e-2. Until 2026-09-15 the gate keyed this on a measurement no solver recorded and thresholded the residual anyway; it now reads `divergence_form`, D-55 item 7.)* | < 1e-6 Ha | DERIVED | [F1]; D-44, D-55 |
| **G2.5** | **Two-solver agreement.** The same Hamiltonian solved by Chebyshev-filtered subspace iteration and by LOBPCG, the latter preconditioned in the similarity frame `f⁻¹·K·f` since D-56 (before that it stagnated on the molecular grids, known-open O-15, closed 2026-09-14). | \|Δε_i\| < 1e-8 Ha for all occupied states (measured ≤ 2.8e-9) | EXACT | [F1], [F4]; D-56 |
| **G2.6** | **Forces against finite differences.** Each Cartesian force component compared with a central difference of the total energy at ±0.001 bohr. | max \|ΔF\| < 1e-4 Ha/bohr | DERIVED | [D3], [D17] |
| **G2.7** | **Egg-box error.** Rigidly translate the whole molecule through one grid cell in 8 steps and record the variation in total energy and in force. **This is the most dangerous silent failure in a real-space code**: the error is smooth, small, has no physical signature, and biases every geometry and every force in the corpus in the same direction. Measured (2026-09-15, D-58): atoms ≤ 1e-11, H₂⁺ R = 2 1.6e-5, H₂ R = 1.4 2.9e-5 Ha/atom — under the threshold on every scenario since D-58 (A-3 closed). | ΔE < 1 meV/atom (3.67e-5 Ha/atom) and Δ\|F\| < 1e-3 Ha/bohr | EMPIRICAL | [D16], [D17], [D18]; D-58 *Restated 2026-09-16 (D-64):* the eight probes solve with `derive_grid=False` on the resolved grid, because a derived lattice is anchored on the first nucleus and would move with the molecule. |
| **G2.8** | **Translational and rotational invariance.** Σ_a F_a and Σ_a r_a × F_a. | \|ΣF\| < 1e-5 Ha/bohr; \|Στ\| < 1e-5 Ha | EXACT | — |
| **G2.9** | **Energy term closure.** The sum of the recorded decomposition (T_s + E_ext + E_H + E_xc + E_NL + E_ii) must equal the reported total to round-off. | < 1e-12 Ha | EXACT | [A2] |
| **G2.10** | **Mixed-precision audit.** Every scenario is re-run in full float64 on CPU at the same grid. | \|ΔE_tot\| < 1e-6 Ha/atom; \|Δn\|₁ < 1e-5 e; \|ΔF\|∞ < 1e-5 Ha/bohr. A breach *fails the run*, it does not merely warn. | DERIVED | [D13], [I6], [I7], [I8] |

---

### Tier 3 — Numerical convergence. Run once per scenario, archived with the scenario.

| ID | Gate | Threshold | Kind | Source |
|---|---|---|---|---|
| **G3.1** | **Grid-spacing convergence.** Restated 2026-09-13 (D-48): the ladder **refines** — three rungs at ratio 1.25 downward from the production spacing on the resolved grid (`derive_grid=False`, D-42), never coarsening — and the verdict is the ground-eigenvalue difference between the two finest rungs per atom. The Richardson estimate of the remaining error, the fitted order and the monotonicity of the sequence are recorded beside the verdict and never thresholded: a power law fitted to the coarse end and evaluated at zero is not a limit. Non-interacting scenarios only; the self-consistent ladders are measured once and recorded in `scripts/interacting_ladder.py` and the ladder table of `02_STATUS.md` Part F, so the gate reports SKIPPED there (budgeted, not deferred). ~130 s per molecule, `full` profile only. | \|E(h₂) − E(h₃)\| < 1 meV/atom (3.67e-5 Ha/atom) over the two finest rungs h₂ = h_prod/1.25, h₃ = h_prod/1.5625 | DERIVED | [D6], [D9]; D-48 |
| **G3.2** | **Domain size convergence.** Increase the mask radius by 2 Å. | \|ΔE\| < 1 meV/atom; the highest occupied orbital must have decayed to < 1e-5 of its maximum at the boundary | DERIVED | [D5] |
| **G3.3** | **SCF convergence criterion.** Declared converged only when **both** \|ΔE\| < 1e-8 Ha and ‖Δn‖₁ < 1e-6 e over consecutive iterations. Two criteria, because either alone can be satisfied by a stalled iteration. The gate reads the loop's stop reason and the final \|ΔE\|, ‖Δn‖₁ and reports the fraction of the tolerance used. | as stated (fraction ≤ 1.0) | DERIVED | [F12] |
| **G3.4** | **Initial-guess independence.** The same system converged from three starts — the superposition of hydrogenic 1s densities (default), a diffuse and a compact variant (`INITIAL_GUESSES`) — must reach the same fixed point (two extra solves; `full` profile). Mixer choice never changes the fixed point. | \|ΔE_tot\| < 1e-7 Ha between all three | EXACT | [F13] |
| **G3.5** | **Stencil-order independence.** Production order 2p = 8 against 2p = 12 at the production spacing. | \|ΔE\| < 1 meV/atom | DERIVED | [D2] |

---

### Tier 4 — Literature reproduction. The hard gates.

**Rule, from [B4] and [B5]: absolute total energies are never compared across codes**, different
pseudopotentials, discretisations and core treatments making them incommensurable — except on the
all-electron path of G4.7, where neither side carries a pseudopotential.

| ID | Gate | Threshold | Kind | Source |
|---|---|---|---|---|
| **G4.1** | **Atomic eigenvalues vs the pseudopotential file.** Each ONCV/PseudoDojo file ships the reference all-electron valence eigenvalues it was fitted to. An isolated-atom SCF must reproduce them. This validates the projector implementation *before any molecule is attempted* — the single highest-value gate in the suite per unit of effort. | \|Δε_valence\| < 1e-3 Ha | EMPIRICAL | [E1], [E2], [E3] |
| **G4.2** | **Bond lengths.** r_e for H₂, LiH, N₂, CO, H₂O, CH₄ at PBE against published grid-code values. | \|Δr_e\| < 0.005 Å. (Our target is *agreement with PBE*, not with experiment. The "0.006–0.014 Å learned vs 0.04–0.05 Å semi-local" contrast once quoted here is **withdrawn**, [A17] corrections of 2026-09-16; what stands is 0.012 Å (CCse21) and 0.014 Å (LMGB35) for Skala.) | EMPIRICAL | [A17], [D5] |
| **G4.3** | **Binding energies, cross-code.** Same molecules, computed in Octopus [D5] at matched settings (same ONCV file, same functional, same spacing, same box). Octopus is chosen because it shares this discretisation family, making it the only code whose numbers are directly commensurable. | \|ΔE_bind\| < 10 meV | EMPIRICAL | [B4], [B5], [D5] |
| **G4.4** | **Dissociation curve, ε-measure.** The H₂ curve from 0.5 to 4.0 Å against the Octopus reference, scored with the dimensionless measure of [B5]: ε = sqrt( ⟨[E_a − E_b]²⟩ / sqrt(⟨[E_a−⟨E_a⟩]²⟩⟨[E_b−⟨E_b⟩]²⟩) ). | **ε ≲ 0.06** (the "excellent agreement" threshold of [B5]); ε ≲ 0.2 is "good" and triggers investigation, not acceptance | EMPIRICAL | [B5] |
| **G4.5** | **Self-interaction error, reproduced deliberately.** H₂⁺ is a one-electron system: the exact E_xc cancels the Hartree self-repulsion exactly, and a semi-local functional does not. Restated 2026-09-14 (D-55 item 2) as the one sign that is textbook: at a stretched geometry (R ≥ 6) `E(H₂⁺)` at a semi-local functional must lie **below** `E(H)` at the same functional and grid rule (one extra solve). The magnitude is recorded (`delocalisation_error_D1_8`) and never thresholded (D-20). A solver that gets the sign *right* has a Hartree or XC bug. | `E(H₂⁺, R ≥ 6) − E(H) < 0` | EMPIRICAL | [A9], [C1], [A17]; D-55 |
| **G4.6** | **Atomization energies** (blocked on Increment 4 / spin). Closed-shell subset of W4-11 [B2]. | MAE < 0.1 eV against published PBE values, per-molecule tolerance recorded | EMPIRICAL | [B2], [B3] |
| **G4.7** | **All-electron agreement with published values.** The solver's totals against NIST SRD 141 LDA (VWN) for H, He⁺, He, transcribed with provenance in `cdft.reference.literature`; where NIST has no row (PBE; H₂) against the PySCF cc-pV5Z totals of `cdft.reference.computed` (`COMPUTED_ORACLE`, D-29), reported with their basis-set uncertainty beside the difference. | \|ΔE\| < 1e-3 Ha at a matched functional (measured He LDA 1.7e-5 at h = 0.25) | EMPIRICAL | NIST SRD 141, doi:10.18434/T4ZP4F; `scripts/pyscf_reference.py` |
| | *Note:* these are targets for the *functional*, never accuracy checks on the solver — the NIST LSD hydrogen energy is −0.478671 against an exact −0.5, and that 0.021 Ha gap is the self-interaction error this project exists to measure. Where no published value at the run's functional and basis has been transcribed the gate reports SKIPPED with that reason (O-12). | | | |
| **G4.8** | **Radial reference agreement.** The three-dimensional solver against `cdft.reference.radial` (non-interacting) or `cdft.reference.radial_ks` (self-consistent LDA/GGA atoms, D-54), independently-implemented solvers: separated rather than discretised in 3-D, on a logarithmic grid that resolves the cusp directly without the transform of D-35, closed by direct banded diagonalisation. The radial KS oracle reproduces NIST SRD 141 to 4e-7 Ha and shares only the functional objects (validated by G0.6) with the 3-D path. | \|Δε\| < 1.5936e-3 Ha | DERIVED | D-36, D-54; separation of variables |
| **G4.9** | **Two-centre reference agreement.** The multi-centre counterpart of G4.8: H₂⁺ against `cdft.reference.two_centre`, by separation in prolate spheroidal coordinates, at **any bond length**. This is what turns a dissociation scan from a curve with one validated point into a curve validated everywhere. | \|Δε\| < 2.5e-3 Ha | DERIVED | D-41; Madsen and Peek, *Atomic Data* **2**, 171 (1971) |
| | *Note:* the threshold is deliberately looser than G4.8's chemical accuracy: the measured multi-centre error is 1.0e-3 Ha at R = 2 and 2.0e-3 at R = 1.4 (O-13, A-1), so a gate at 1.5936e-3 would fail on a defect already recorded and scheduled. A gate that catches *new* breakage sits above the known error and moves down as it is attacked. | | | |

**Why G4.5 is a gate and not a bug report.** Self-interaction error is not a defect in this baseline
but the most valuable region of the corpus — the state of the art still has its largest GMTKN55
subset error there ([A17], [C1]; `01_PROJECT.md` Part C) — so the gate exists to make it reproducible
rather than to forbid it. Consequence: **occupations are floating point from Increment 1** (D-11).

---

### Tier 5 — Anti-silent-failure. Process gates, not physics.

| ID | Gate and requirement |
|---|---|
| **G5.1** | **Provenance.** Every record embeds: git commit SHA, dirty-tree flag, full config hash, package versions (torch, numpy, libxc, h5py), the contract's device block (`device`, `device_name`, `device_capability`, `torch_cuda_version`, `cuda_driver_version`, `deterministic_algorithms`, `cublas_workspace_config`; `gpu_peak_bytes` required on a completed CUDA run), precision policy, and wall-clock timing. A GPU record is distinguishable from a CPU one by this block alone. *Restated 2026-09-15 (D-60).* |
| **G5.2** | **Determinism.** Same config, same seed, same device → total energy reproducible to < 1e-9 Ha. `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` whenever the device is CUDA (D-61); the repeat runs on the device recorded in the artifact's provenance and the detail names the leg (CPU/GPU), both runs' deterministic mode and whether the repeat was bitwise. *Restated 2026-09-15.* *Restated 2026-09-16 (D-69):* the gate runs **two strict solves of its own** (A and B, `solves = 2`, cross-check off, geometry cache bypassed) and thresholds \|ε_A − ε_B\| at 1e-9; the primary solve of every profile runs with deterministic mode **off**, and the primary-vs-A difference is recorded (`primary_vs_strict`), never thresholded; `strict_pair_wall_s` records the pair's cost. *Kind restated 2026-09-18 (B3):* **EXACT** — the reference is the identity (the same configuration solved twice, expected difference 0) and the 1e-9 tolerance only absorbs float64 reduction-order noise (2.7e-13 measured, D-61); it follows from no convergence parameter, so it is not DERIVED. Catalogue and `GateSpec` both say EXACT. |
| **G5.3** | **Gate report attached.** Every record carries the full gate table with measured values, not just verdicts. A value that passed but sits within 10 % of its threshold is flagged `MARGINAL` in the report. |
| **G5.4** | **Corpus reader enforces status.** The default corpus iterator yields only `status == VALID`. Reading INVALID records requires an explicit keyword argument. |
| **G5.5** | **No default-silent fallbacks.** If the primary eigensolver fails, if mixing falls back a level, or if a tolerance is relaxed, it is recorded in the run artifact and downgrades the record's status. Fallbacks are allowed; hidden fallbacks are not. |
| **G5.6** | **Non-zero exit.** A batch run exits non-zero if any record failed a gate. Unattended runs must be able to fail. |
| **G5.7** | **Scope boundary.** The package `src/cdft/` contains no machine learning. CI asserts, by a static `ast` scan of every module under `src/cdft/` and of the solver modules at the repository root (D-57, D-72), that none imports a training framework or optimiser (`torch.optim`, **all of `torch.nn`** — a trained functional is a `Module` with parameters, and evaluating one needs none of it — `sklearn`, `optax`, `lightning`, `pytorch_lightning`, `keras`, `tensorflow`, `jax.example_libraries`) and that no function named `fit`, `train`, `train_step`, `training_step` or `backward_pass` is defined. The scan does not inspect `XCFunctionalProtocol` implementations; that they expose `evaluate` only is a contract rule, not a gate. Evaluating an externally-supplied functional is permitted; producing one is not. See `01_PROJECT.md` (Part B, Goals) §3.4. *Restated 2026-09-18, in the gate review of D-75, to what the scan enforces.* |
| **G5.8** | **Gate catalogue consistency — the gate that gates the gates.** Cross-checks three descriptions of what a gate is: the table in this document, `cdft.gates.catalogue`, and the evaluators wired into the runner. Counts four kinds of disagreement — claimed implemented with no evaluator; an evaluator for a gate marked deferred; an evaluator the catalogue does not define; a gate a scenario names that the catalogue does not define — plus any deferral with no named increment. Gates enforced outside the evaluator tuples (G5.3, G5.4, G5.6) are listed with *where*, and a listed location that no longer resolves is itself a disagreement. Threshold zero disagreements, EXACT (D-40, D-37). |
| **G5.9** | **Sync/launch budget.** The census of `scripts/sync_census.py` (host reads, implicit-sync operators, kernel-launching operators), run on the CPU on two cheap probes — `h_atom` (one CheFSI solve) and a coarse self-consistent helium (h = 0.5, 8-bohr box, `derive_grid=False`) — may not exceed the budgets recorded beside the gate: Python reads and implicit syncs per iteration at loop sites, launches per solve, each with 10 % headroom (per-iteration headroom capped at 0.5, so one new read per iteration fails). A reintroduced `.item()` in a loop fails the build. Reads and implicit syncs on tensors that live on the host by design (`HOST_SIDE_READS`: the Lanczos bounds, two mixer scalars, the memoised Gauss–Legendre rule built on a CPU tensor on every device) are reported apart and never count (D-82). *Added 2026-09-16 (D-70); host-side implicit sites 2026-09-19.* |

---

### Tier 6 — Gates created by the cusp factorisation

The transform of D-35 changes what some quantities *mean*. This tier is numbered separately so that
removing the transform would remove a tier rather than leave orphaned rows in Tiers 0 to 5.

| ID | Gate | Threshold | Kind | Source |
|---|---|---|---|---|
| **G6.1** | **Hellmann–Feynman forces including the cusp-factor terms.** The transform makes the weight `f²` depend on the nuclear positions, so differentiating the energy with respect to a nucleus picks up terms that do not exist on the untransformed path. A force expression that omits them is wrong in a way no force gate on the untransformed path would catch. | as G2.6, once forces exist | DERIVED | D-35 |
| | *Note:* **DEFERRED to I7** (forces), with G2.6 and G2.8. Recorded now so that the extra terms are not discovered during the forces increment. | | | |

---

### Gate-to-increment map

An increment is not closed until its gates pass **on both the CPU-float64 and the GPU-mixed paths**.

| Increment | Gates that must pass to close it |
|---|---|
| I0 | G5.1, G5.3, G5.6, G5.7 |
| I1 | G0.1–G0.4, G0.8, G1.2, G1.3, G1.4, G1.6, G2.4, G2.5 |
| I1b (cusp-factorised path) | G0.9, G0.12, G1.7, G1.11, G1.12, G1.13, G2.7, G3.1, G3.2, G3.5, G4.8, G4.9, G5.2, G5.5, G5.8; G4.7 once I2 supplies a functional |
| I2 (closed) | G0.5, G0.6, G1.1, G1.8, G1.9 |
| I3 (closed) | G1.5, G1.7 (interacting form), G2.1, G2.2, G2.3, G2.9, G3.3, G3.4, G4.7, G4.8; G4.5 brought forward from I8 (D-55) |
| I4 spin (D-51) | G4.6 |
| I5 pseudopotentials | G0.7, G4.1 |
| I6 GPU (first half closed) | G5.1, G5.2 GPU legs; G2.10 with the float32 hot path |
| I7 forces | G2.6, G2.8, G6.1 |
| I8 molecular benchmarks | G4.2, G4.3, G4.4 |
| I9 corpus generation | G5.4 |
| Throughput work | G5.9 added (D-70); G5.2 restated as the strict pair (D-69) |

### Reporting format

```
GateResult(gate_id="G2.7", name="Egg-box error", verdict=PASS|MARGINAL|FAIL|SKIPPED,
           measured=2.14e-5, threshold=3.67e-5, kind=EMPIRICAL, citation="D17; D16",
           detail={...})
```

`SKIPPED` requires a reason and is itself reported, so a gate silently not running is impossible (the
one deliberate absence is an oracle gate while the oracles are off: dropped from the plan by design,
and the record says so in `measurements["oracles"]`, D-76); a value that passed within 10 % of its
threshold is flagged `MARGINAL` (G5.3).

---

## Part D — The validation ladder

Gates passing is only as strong as the gates, and most of Tier 1 compares against analytic limits
that exist for one- and two-electron systems and nowhere else. This is the ladder of independent
checks, what each is worth, and what none of them can see.

| # | Reference | Independent of | Accuracy of the *reference* | Covers | Gate |
|---|---|---|---|---|---|
| 1 | closed forms — harmonic well, box, hydrogenic spectrum | everything | exact | model systems, one-electron atoms | G1.2, G1.3; G1.13 (closed-form ∫f²) |
| 2 | dense diagonalisation of the matrix-free operator | the eigensolver | exact | the operator itself | G1.6 (8.9e-14) |
| 3 | second eigensolver (LOBPCG) | the production solver | — | the eigenproblem | G2.5 (4.2e-9) |
| 4 | radial solver — separation, log grid, banded diagonalisation | the 3-D grid, the stencil, the halo, the cusp transform, the eigensolver | 4.5e-12 (Z=1) … 8.4e-10 (Z=8); radial KS 4e-7 vs NIST | **any** spherical potential, **any** functional | G4.8 |
| 5 | two-centre exact — prolate spheroidal separation | everything | 2.2e-9 vs Madsen and Peek | H₂⁺ at any R | G4.9 |
| 6 | libxc, pointwise | the transcription of the functionals | 1e-10 | the XC layer | G0.6 |
| 7 | Gaussian basis / literature — NIST SRD 141, PySCF cc-pV5Z | the whole discretisation | published / basis-set uncertainty | absolute energies | G4.7 |
| 8 | Octopus, matched settings | the implementation, not the method | — | molecules | G4.3 (I8, O-2) |

**Why rung 4 is the important one.** For a spherically symmetric `v(r)` the Kohn–Sham equation
separates exactly, and one dimension changes everything: a logarithmic grid resolves the cusp
**directly**, with no factorisation and no regularisation, which is what makes it an independent
check of D-35; the eigenproblem is closed by direct banded diagonalisation, so there is no filter and
no convergence criterion to get wrong; and it costs about a second where the same resolution in three
dimensions would be ~10¹¹ points. It is validated against closed forms before being trusted
(`tests/test_radial_reference.py`). Against the three-dimensional solver at h = 0.25 in a 14-bohr
box: H −0.4999796920 against −0.5 (**2.03e-5**), He⁺ −1.9999999999 against −2.0 (**6.73e-11**). That
difference *is* the three-dimensional discretisation error, measured against something that does not
share it, and the oracle keeps working once the functional arrives, where the analytic answers stop.

**What the ladder still cannot see.**

1. **Non-spherical systems have no rung 4.** H₂ and every molecule fall back to rungs 5 (H₂⁺ only),
   7 and 8; the multi-centre penalty is measured directly instead (Part B §4).
2. **Nothing here validates the functional.** Every rung checks that the chosen equation is solved
   correctly; whether LDA, PBE or r2SCAN is *right* is what the project exists to measure, which is
   why the diagnostics of `01_PROJECT.md` (Part C) are a category apart from the gates (D-20).
3. **Density integrals are not covered by G1.5**, which holds by construction whatever the rule is
   worth; G1.13 against the closed form of ∫f² is the check that means something (A-9, O-19).
4. **The GPU path is verified in float64 only.** The card reproduces the CPU to 2.05e-13 Ha and G5.2
   passes on it; G2.10 has no float32 path to audit until the float32 hot path is wired.

For spherical, all-electron, non-interacting systems the results are sound at a level *measured
against an independent method*; for two-centre systems the check is analytic and the accuracy an
order and a half worse; for interacting systems the radial KS oracle is the standard, and nothing
beyond atoms has an independent reference.

---

## Part E — Code rationale, module by module

The reasoning that does not live in the code: modules keep a one-line docstring and a `D-nn` pointer,
and this Part keeps the *why* (D-73) — purpose, the decisions a future editor must not undo, the
gotchas, the switches. Where a decision has a `D-nn` the log carries its measurements and this Part
carries the consequence. Gate rationale stays in the gate; `tests/`, `test_suite.py`, `contract.py`
and `scripts/` keep their own prose.

### `cdft/__init__.py`, `constants.py`, `structure.py` — the package root, units, geometry

- `contract.py` sits at the repository root, not in the package: it is a specification the
  implementation is written against (D-15). `CONTRACT_VERSION_EXPECTED` ("1.6.1", D-80) is checked
  against the version a record carries; a record stamped with an unimplemented contract is
  unreadable. "Classical" means non-neural: no ML under `src/cdft/` (D-24), enforced by G5.7.
- Hartree atomic units throughout, converted only at the configuration and reporting boundaries;
  CODATA 2018 at full published precision, because G2.9 closes the decomposition at 1e-12 Ha. `C_X`
  and `LIEB_OXFORD_FACTOR` are computed, not transcribed [A8], [A10], [A16]; `atomic_number` raises
  on an unknown symbol.
- `structure.py` is the only place a length in Angstrom may enter and `units` is mandatory (a geometry
  wrong by 1.889 converges beautifully); `positions_tensor` stays float64 whatever the hot dtype,
  float32 putting ~1e-7 bohr into `v_ext` that no SCF convergence removes.

### `cdft/precision.py` — the precision policy and the device seams

Applies Part A §6 and owns the device seams that are not physics.

- `hot_dtype` is float64 on the CPU whatever the configuration says, making the CPU path G2.10's
  audit path by construction, and float64 on CUDA until the float32 hot path lands (D-60): a record
  claiming float32 for a float64 run violates G5.1. `policy_record` is written per run.
- `as_hot` / `as_accumulate` are unreferenced on purpose: the declared casting seam for the float32
  hot path (`02_STATUS.md` Part C, D-3).
- Device behaviour is decided in one place, `device_policy.py` (D-70): a `DeviceProfile` becomes
  `DerivedSettings` (cache bytes, graph memory, trsm chunk, graphs, concurrent solves) written into
  every record, every feature off on the CPU and TF32 asserted off everywhere. On CUDA the divergence
  stencil and the Chebyshev recurrence may run as fused kernels chosen by a first-use probe (D-65).
  G5.9's budgets are that policy's audit: recorded 2026-09-16 on the CPU, `h_atom` 7 CheFSI
  iterations at 9/7 reads and 16/7 implicit syncs per iteration with 34 841 launches, and
  `he_scf_probe` 12 SCF iterations at **80/12 Python reads and 118/12 implicit syncs per iteration,
  127 429 launches** (D-70). Geometry-only objects are cached **by value**, `CACHE_VERSION` included,
  with a `bypass()` so G5.2 and G5.9 measure a fresh build (D-66); the CPU body stays eager and
  pinned bitwise, re-associated arithmetic differing at ~1e-12 relative.
- `solve_triangular_wide` chunks above `CUDA_TRSM_CHUNK = 2¹⁸` columns **on CUDA only** (the cuBLAS
  `trsm` cliff, D-62), chunking being bitwise there and not on a CPU BLAS;
  `scripts/probes/trsm_probe.py` re-measures the constant after a driver or torch upgrade.

### `config.py` — numerics loading and the named presets

- Canonical at the root since D-57; nothing here may name a species (Q1.5). Loaders reject unknown
  keys and enum values and print the alternatives — a mistyped `spacing` silently ignored is a run at
  the default grid reported as a run at the requested one. `numerics_to_dict` is exactly what
  `fingerprint` hashes, so a configuration and its `config_hash` cannot disagree.
- **`MODEL_SYSTEM_NUMERICS` uses `spacing = 0.20`**: G1.2's ten-level error falls 4.05e-6 → 7.06e-7
  between h = 0.25 and 0.20 at a measured order of 8.04, for 133 s per solve against 51 s. Its
  `residual_tol = 1e-8` sits an order below G2.4's 1e-6, since at the contract default the solver
  stopped at 9.5e-7 and G2.4 read MARGINAL — a gate describing the stop criterion instead of
  checking it.
- **`ALL_ELECTRON_NUMERICS` is a placeholder, not a choice**: box and spacing are derived per
  scenario and recorded as overrides (D-42, G5.5). It holds the single-centre `Z = 1` values so a
  caller who bypasses the derivation still gets a sane grid, the contract default of 0.3 being in
  neither regime.

### `config.py` setups and `cdft.grid.grid_rule` — resolution levels (D-73)

- One `Setup` per registered scenario with levels `draft`, `standard`, `fine`, `reference`.
  **`standard` *is* the production rule** (D-42/D-53/D-64) and the only level comparable with the
  golden values, the accuracy ledger and the known-open registry.
- `run.py --setups` lists them, `--res fine` runs one; the default numerics is the scenario's own
  preset. A non-standard level applies through `cdft.grid.grid_rule(...)` and is recorded as
  `grid_overrides["grid_rule"]` (G5.5), so its numbers cannot be mistaken for production ones.
- The API: `config.numerics_for(scenario_id, level, **overrides)` returns `(numerics, grid_rule
  targets)`, `numerics_for_spec` does the same for a `--scenario-file` scenario that has no setup,
  and `describe_setups()` renders the table `--setups` prints.
- **A model-system level must divide the box edge exactly**, or `UniformGrid.from_config` snaps `h`
  and G5.5 flags the override — hence per-scenario model levels rather than one global ladder.
- **Cost warning:** `reference` on `h2plus_R8_lda` (181³ = 5.9 M points), `reference` on
  `h2_R1.4_lda/pbe` (144³ = 3.0 M) and `fine` on `h2plus_R8_lda` (3.0 M) do not fit a 6 GB card.
- `run.py` refuses `--res` (or a field override) together with an explicit `--numerics` preset or
  YAML file and exits **2**: a preset fixes every numerics value, so mixing the two would silently
  discard one of them (D-73).

### `cdft/grid.py` — the real-space grid and the sizing rules

- **The sizing constants are derived and the two regimes are opposite** (D-42, which carries the
  sweeps): `COULOMB_HALF_BOX_PER_Z = 16.0` because a single centre is box-limited by
  `f²(R) = e^{−2ZR}`; `POINTS_PER_EDGE = 33` is a *point count*, not a spacing, because refining `h`
  on a constant φ only accumulates round-off while below ~11 points per edge He⁺ and G1.11 collapse;
  `VACUUM_PADDING = 12.0` with `MOLECULAR_SPACING = 0.25` is the molecular regime, where the box buys
  nothing and convergence is monotone only below h = 0.5. D-53 adds the derived vacuum margin, floor
  7 bohr.
- **Box snapping** puts the origin exactly on a grid point, because the on-site regularisation of
  `−Z/r` applies only where a nucleus coincides with one: unsnapped, a refinement ladder alternates
  between the two treatments, hydrogen comes out *worse* at the finest spacing, and Richardson
  extrapolation is meaningless. Under `ODD_REFLECTION` an edge `L` holds `L/h − 1` interior points,
  making the discrete spectrum the analytic one.

### `cdft/operators/laplacian.py` — finite-difference weights

- **Weights in exact rational arithmetic** (Vandermonde with `fractions.Fraction`): a transcribed
  table puts a hand-copied integer in the innermost loop, wrong everywhere, converging smoothly and
  passing everything that does not measure its *order* — which G0.1 does.

### `cdft/operators/divergence.py` — the staggered divergence form (D-38)

- `A_w φ = −(1/2f²) Σ_d D_dᵀ[f²_face,d D_d φ] + W φ`: differentiate onto faces, weight there, return
  with the **exact transpose**. Self-adjoint in the `f²` measure by construction at any order for any
  weight, positive semi-definite kinetic term, and no checkerboard null space, a staggered stencil's
  nearest terms being `f(i+1) − f(i)`. What the two earlier forms cost is Part B §7.

### `cdft/operators/cusp.py` — the Kato factor

Implements `f`, `∇u`, `W`; the transform is Part B §1–2. With no nuclei the operator is bit-for-bit
the untransformed one, and `f` is evaluated analytically at the exact nuclear coordinates.

- **`_build_potential` cell-averages instead of evaluating pointwise** (D-39), by an order-4
  Gauss–Legendre product rule within `_average_radius_cells = 2` spacings, because a unit vector is
  discontinuous *at* its nucleus and `W` nearby otherwise reads which side a point fell on (Part B
  §5). The point exactly on a nucleus then needs no special case (`⟨ŝ_a⟩` vanishes over a centred
  cell, giving `Z_a² + |rest|²`), and a nucleus *between* points goes through the same code: one
  expression for `W` everywhere, so no branch can be right in one place and forgotten in another.
- **`kinetic_energy` is expanded term by term**: `|∇φ|² − 2φ(∇φ·⟨∇u⟩) + φ²⟨|∇u|²⟩`, not
  `|∇φ − φ∇u|²`, because `⟨∇u⟩` vanishes at a nucleus while `⟨|∇u|²⟩ = Z_a² + |rest|²` does not —
  using `|⟨∇u⟩|²` throws `Z_a²` away exactly as the original zeroing bug did (T = 0.48986 against
  0.5 for hydrogen at h = 0.4).

### `cdft/operators/quadrature.py` — the cusp-aware quadrature (D-54, D-58, D-75)

The module that closed A-9, C1, A-11 and C8; the design and its measurements are D-54, D-58 and
D-75, the shape is here.

- A smooth **partition of unity** (erf tapers of fixed spatial width), not exact cell integrals with
  a plain sum outside, which left O(h³) at the seam; the angular rule scales with the radius
  (`n_θ = ⌈√(2π) r/h⌉`), since the lumped weights are sums of *signed* interpolation weights and turn
  negative on a coarser rule — the constructor refuses them, stating the `Z·h ≤ 1` requirement.
- Overlapping spheres are split by erf cells of fixed *spatial* width (D-58); capping the spheres at
  the internuclear distance instead squeezed the taper at R = 1.4 and was the whole of the egg-box
  residue (A-3) and the molecular `∫f²` residue (A-11).
- **Two rejected alternatives, kept so nobody retries them** (D-58): Becke cell functions, whose
  transition width is ∝ R, leak into a lune no sphere rule covers (2.4e-6 on a smooth Gaussian at
  R = 2); weighting those cells by the sphere's own erfc bump switches sharply where the ratio of two
  exponentially small numbers is not negligible against `Q` — **2.2e-5** on the same test, against
  the fixed-spatial-width erf cell integrated by the sphere rules (the taper meets the plain sum at
  5.7e-8, the cell never does — D-75).
- **The mass weights of an overlapping geometry take a lumped far rule (D-75).** The plain `h³` sum
  aliases the erf taper shell at `exp(−π²w²/h²) = 5.7e-8` of its content at `w = 1.3 h` (an earlier
  "4e-15" counted the exponent twice); one shell centred on a lattice nucleus cancels that by
  symmetry (4e-11 on H at `Zh = 1/4`), a second centre modulates it and does not (2.66e-8 on H₂⁺ —
  the whole of A-11). So where spheres overlap, the far part of the mass weights is `Q_far[f² ℓ_i]`
  from a temporary lattice refined 2× per axis over the whole box (three one-dimensional transposed
  Lagrange interpolations; 1.0 s at 57³, no solve-time cost, no memory), blended back to the plain
  weight within 6 h of a box face, where one-sided stencils swing the weight by 2× and doubled the
  CheFSI iterations. **Only the mass weights:** `Σ ω_i g_i = Q_far[f² I[g]]` interpolates `g`, and a
  field with a cusp of its own — `v_xc ~ n^{1/3}`, `e_xc` — is interpolated badly where the taper's
  inner tail still carries weight (+1.1e-6 Ha on the H₂ LDA total); `integrate`, `lumped_weights` and
  the GGA gradient term keep the plain far sum, and the record carries `far_aliasing`, the 2.7e-8 the
  two rules differ by on `∫f²`. Rejected: a blend around the spheres, the degree-7 stencil
  interpolating an erf of grid-scale width to 4e-3 and costing 7e-9 wherever `f²` is not negligible.
  The lune — inside sphere `b`, outside sphere `a`, where `β_a Q` had no rule — goes to the sphere
  whose nodes cover it (9e-10 on both molecules). A single centre keeps the plain rule bit for bit.
  Denser angular nodes do not help (0.7 → 0.3 h leaves H₂⁺ at 2.5e-8, and 0.5 h OOMs the 6 GB build);
  what remains is the Gauss rule in cos θ at 0.7 h, ±3e-9.
- `L` is SciPy CSR (torch's `.t().to_sparse_csr()` took 11 s on a 57³ grid against 2 s), and
  `lift_gradient` is minus the divergence of the lift, so the GGA weak form never differentiates a
  cusped field; the 1s Hartree split leaves a smooth remainder for the FFT solver, so `poisson.py`
  and G0.2 are unchanged.

### `cdft/operators/external.py` — external potentials

- **The singularity is regularised, and that is stated**: the coincident point gets the average of
  `−Z/r` over the sphere of one cell's volume, `−3Z/(2r_c)` with `r_c = (3/4π)^{1/3}h`, its error
  measured by `scripts/probes/cusp_factorisation_probe.py`; the threshold is half a spacing, so the
  treatment does not depend on the arbitrary grid origin.

### `cdft/operators/poisson.py` — the open-boundary Hartree solver

- The Coulomb-cutoff kernel of [F15] is used through its *analytic* transform, not the DFT of its
  samples, because `1/r` is not band-limited and sampling first aliases at the singularity in a way
  no refinement removes. `pad_factor` ≥ 2 makes the circular convolution the linear one and the
  default is 2.5 (at 2.0 the far field is 1.1e-4 wrong while the energy is exact to 5e-16, G0.2's
  second leg); `charge_outside_cutoff` records the charge outside the valid radius, and `solve`
  refuses a spin-resolved density.

### `cdft/operators/hamiltonian.py` — the matrix-free seam

- `CuspFactoredHamiltonian` is the same seam, so every eigensolver, gate and diagnostic works
  unchanged; `uses_divergence_form` is recorded on every run — a corpus mixing two discretisations
  would be unreadable — and it gates the second convergence criterion in `chefsi`.

### `cdft/eigen/measure.py` — the inner product an eigensolver works in

- Both solvers work in an abstract SPD product rather than a hard-coded `h³` sum, because
  `A = f⁻¹Hf` is self-adjoint only in `⟨a,b⟩_w = ∫f²ab` — otherwise the subspace matrix is not
  symmetric and its eigenvalues are the Ritz values of nothing — and because keeping that fact local
  meant adding the transform changed no line of either solver. The uniform case is dispatched
  separately for speed; a zero or negative weight is refused.

### `cdft/eigen/rayleigh_ritz.py` — orthonormalisation and the Rayleigh–Ritz step

- Unconditionally float64 [I8] (G0.3), on a block tiny against the grid. `orthonormalise` falls back
  from Cholesky to an eigendecomposition with small eigenvalues dropped and *reports* it by returning
  fewer vectors, never by silently returning a non-orthonormal basis; there is one subspace matrix
  from one operator, `⟨a|A|b⟩_w` built from `apply` being the weak form itself since D-38.

### `cdft/eigen/chefsi.py` — the production eigensolver

Chebyshev-filtered subspace iteration [F1]–[F3]: needs only `H·ψ`, nearly all work is a dense product
on a block, and there is no preconditioner to tune — which matters for an unattended instrument.

- **The graphed filter and the carried bounds** (D-67, D-68, both CUDA-side and both recorded): one
  filter step per block shape is captured as a CUDA graph whose scalars are static buffers refreshed
  before each replay, so the captured body is bitwise the eager filter and every capture is validated
  once against it; `SpectralBoundsHint` reuses the previous iteration's Lanczos interval widened by
  `max|Δv_eff|` — a rigorous Weyl enclosure, only the diagonal potential changing — with seven
  triggers that force a fresh estimate. Switches: `CDFT_CUDA_GRAPHS=0`, `CDFT_LANCZOS_REUSE=0`.
- **The second convergence criterion is gated (D-44).** "Ritz values have stopped moving" earns its
  place only on the weak-form path, where the eigenvalue comes from an exactly symmetric Galerkin
  matrix while the residual is measured with an `apply` symmetric only to the stencil's order (He⁺ is
  right to 8e-11 Ha with a collocation residual of 1e-2). Applied indiscriminately it is a premature
  stop that looks like convergence — the harmonic well stopped at 2.6e-6 against a configured 1e-8
  and reported `converged`. Hence the gate on `uses_divergence_form`, `eigen_stop_reason` recorded
  (G5.5). **The general rule: a criterion justified for one operator must be gated on that operator,
  not merely documented as being for it.**

### `cdft/eigen/lobpcg.py` — the permanent oracle

- It **is** preconditioned (Teter–Payne–Allan [F7]): unpreconditioned it stalled at 3e-4 after 200
  iterations and G2.5 was skipped, and a preconditioner only changes the *path* to the fixed point,
  so the oracle stays an oracle.
- **It acts in the similarity frame `f⁻¹K(f·)`** (D-56, closed O-15/C5), because `K` models the plain
  Laplacian — the operator on `ψ = fφ`, not on `φ` — and `r_ψ = f r_φ` exactly; in the transformed
  frame the filter was measurably *harmful* on both molecules. All three frames stay selectable
  (`precondition="similarity" | "transformed" | "none"`).

### `cdft/scf/noninteracting.py` — the non-interacting solve

- **`E_ext` is the band-energy complement** (`band − kinetic`), the direct integral kept beside it as
  `external_direct` with `external_method` saying which is which (A-6): the direct form sums an
  integrand with an integrable `1/r` pole, the total never needs it (D-52 item 2), and the gap
  between them measures the quadrature of a cusped density (A-9). Permanent on both sides of I3.
- **`cusp_weight_*` and `density_amplitude_ratio`** (G1.13, A-9) record `exact/grid`, the factor by
  which the density is wrong *at every point*, which G1.5 cannot see and the eigenvalue — a Rayleigh
  quotient of a constant φ, exact for any weights — cannot either.
- Also recorded: `eigen_stop_reason`; `nucleus_offgrid_bohr`, because the on-site regularisation of
  `−Z/r` applies only where a nucleus coincides with a point, so a scan whose bond length is not a
  multiple of the spacing shows scatter indistinguishable from physics; and `hermiticity_error`, on
  *random* vectors, a smooth probe having decayed where a boundary bug lives.

### `cdft/scf/step.py`, `loop.py`, `mixing.py`, `occupations.py`, `solve.py` — self-consistency

- **`KohnShamStep` is the contract's `SCFStepProtocol` and is pure** (D-49): `step` and `residual`
  rebind nothing on `self` and return new tensors, which a test asserts by calling twice. They are
  the pieces a differentiable-SCF project would need, which D-47 keeps out of scope. **`solve.py` is
  the one door** (D-53): every script, gate and test solves through `solve_scenario`, so a scenario
  cannot be solved on the wrong path.
- Occupations are floating point from the first increment (D-11), including the closed-shell case:
  the fractional-electron region is the most valuable part of the corpus ([C1], [A17]) and must be
  expressible without a schema change. Smearing [F18]–[F20] arrives with spin polarisation and is
  deliberately not stubbed in, the bracketing search it needs being a documented source of silent
  failure.
- **The loop is imperative and boring on purpose**: four Chebyshev filter steps per iteration from
  the previous orbitals, *without* the D-44 Ritz-stability stop, because the density needs
  eigenvectors and not just eigenvalues (D-55 item 8); periodic Pulay [F9] by default with the
  fallback ladder of [F12] — Pulay → damped → linear → level shift — every descent recorded in
  `scf_fallbacks` (G5.5), and Kerker [F11] implemented and off. Convergence needs **both**
  `|ΔE| < 1e-8` and `‖Δn‖₁ < 1e-6` (G3.3), and `INITIAL_GUESSES` exists for G3.4.

### `cdft/xc/` — the exchange–correlation layer

- **One base class, autograd potentials**: a `SemiLocalFunctional`'s only physics is its energy
  density and `evaluate()` differentiates it, so there is no hand-written potential to get wrong and
  G0.5 compares exactly the two things the design relates. The spin-polarised form is the *only* form
  (a restricted call is `n/2, n/2`), so I4 changes nothing here; thresholds are named constants
  (`DENSITY_THRESHOLD = 1e-14`, `SIGMA_THRESHOLD = 1e-40`, polarisation clamped at `±(1 − 1e-15)`)
  and any departure from a published formula is one of them.
- Native: Slater; PW92 [A8] original and "modified" (the modified set is what PBE correlation is
  built on; they differ by 1e-8 in `e_c`); VWN5 [A7]; PZ81 [A9], branches meeting at `r_s = 1` with a
  3.2e-5 discontinuity *as published*; PBE [A4] with the libxc constants (`κ = 0.804`,
  `μ = 0.2195149727645171`, `β = 0.06672455060314922`, `γ = (1 − ln2)/π²`); B88 [A5]; LYP [A6],
  Miehlich form; r2SCAN [A11] a placeholder raising `NotImplementedError` (D-55 item 5).

### `cdft/gates/` — the gate machinery

- `catalogue.py` is the single source of truth for what a gate *is* (D-40), cross-checked against
  Part C and the wired evaluators by G5.8 and `tests/test_gate_catalogue.py`; DEFERRED is a
  first-class verdict with a named owning increment (D-37), and a known failure keeps its verdict and
  loses its power to fail the build through `known_open.py` (D-43). The ladder gates G2.7, G3.1, G3.2
  and G3.5 skip interacting scenarios with a stated reason — a ladder of four-minute SCF solves
  cannot run at solve time, and `scripts/interacting_ladder.py` is where that measurement lives.

### `cdft/io/provenance.py` and `cdft/io/hdf5.py` — the record and the corpus

- Provenance is G5.1's list; `git_dirty` is recorded rather than forbidden, the git calls fail
  non-fatally so a host without git still produces a record, and an absent package is recorded as
  `"absent"`, which differs from nobody having written it down.
- The corpus is append-only and its default iterator yields only trusted records (G5.4); the filter
  is `TRUSTED_STATUSES`, not a comparison against `VALID`, because an earlier version dropped every
  `MARGINAL` record — a pass. A reader that is too strict loses data as silently as one too lax.

### `physics_config.py` — the scenario registry

- Canonical at the root (D-57); never contains a grid spacing. Sixteen entries, nine in
  `QUICK_PROFILE_SCENARIOS`; a second scenario under an existing id raises, the id being written into
  every record. Reference values are analytic or cited, never "approximately known": where one is
  wanted but not transcribed the entry is *absent* and the gate reports SKIPPED with a reason (D-29,
  and O-12's PySCF `COMPUTED_ORACLE` where the literature has no total at the run's functional).

### `cdft/reference/radial.py` — the radial oracle (D-36)

- **`r_min` sets the accuracy, not the point count** — the opposite of what a 3-D grid trains, and
  measured rather than guessed, the Dirichlet condition imposing `u(r_min) = 0` where the true
  `u ~ r R(0)`: hydrogen gives 2.0e-7 at `r_min = 1e-7`, **unchanged by tripling the points**, and
  4.5e-12 at 1e-12, while raising `n_points` costs cubically and does not help.

### `cdft/reference/radial_ks.py` and `computed.py` — the interacting oracles

- `solve_radial_ks(Z, N, functional)` is the self-consistent radial oracle: 1s shell, log grid, the
  Hartree term by cumulative Simpson integrals (a trapezoid left 3e-5 Ha in the helium total), the
  GGA potential in radial form, linear mixing. It shares *only* the functional objects with the 3-D
  path (validated by G0.6), so an agreement is a statement about the solver: NIST SRD 141 to 4e-7 Ha
  at LDA, and the PBE reference where NIST has none; ~30 s per atom.
- **Trust radius of its density: 1e-4 bohr** (O-25, closed as a stated limit). The Dirichlet node
  `u(r_min) = 0` at `r_min = 1e-12` admixes the irregular `l = 0` solution (`u ~ const`) at relative
  amplitude `~r_min / r`, so `n e^{2Zr}` is low by 4e-6 at 1e-6 bohr and 0.2 % at 1e-9, and by
  < 1e-8 from 1e-4 bohr outward; totals are unaffected at 4e-7 Ha, the charge inside `r_min` being
  `(4π/3) n(0) r_min³`. `figures.py` reads the oracle only from `RADIAL_ORACLE_TRUST_BOHR = 1e-4` and
  fits `ln n` inside it. Removing the layer needs an inner boundary carrying the regular solution
  (`u ∝ r`) instead of zero — not done, the trust radius covering every use.

### `cdft/reference/two_centre.py` — the two-centre oracle (D-41)

- **`_flux_operator` is the whole subtlety.** The coefficient `p` vanishes at the physical endpoints,
  which are *regular singular points, not walls* — `ξ = 1` is where the bonding density lives — so a
  vertex-centred grid drops that coupling and silently imposes `X = 0`, converging to −0.2588 instead
  of −0.6026 and *moving further away* under refinement. On a **cell-centred** grid the outer faces
  land where `p = 0`, so the natural boundary condition is exact rather than imposed.

### `cdft/reference/literature.py` — transcribed published values

- **The rule**: a number here is analytic, computed by an oracle in this package, or transcribed from
  a source named precisely enough to open and check — no fourth category, no "approximately known".
  `LiteratureValue` refuses an entry without a citation or in any unit but Hartree.
- **The NIST values are not exact energies of physical atoms** but exact answers to a different,
  precisely specified problem (Kohn–Sham with VWN, radial, to one microhartree), so the gap between
  the LSD hydrogen energy (−0.478671 Ha) and the exact −0.5 is not a solver error but the
  **self-interaction error of the functional**: H/lda 0.054 Ha (D1.1), H/lsd 0.021 Ha, He/lda
  0.069 Ha with LDA and LSD identical, He⁺/lda 0.139 Ha.
- Sources: `NIST_SRD141` (Kotochigova *et al.*, NIST SRD 141, 2009, doi:10.18434/T4ZP4F, 1e-6 Ha,
  KS LDA/LSD with VWN, non-relativistic, point nucleus, radial); `EXACT_NONRELATIVISTIC` (He
  −2.9037243770341195983110, H and He⁺ closed form); `TWO_CENTRE_TABULATED` (Madsen and Peek,
  *Atomic Data* **2**, 171 (1971): H₂⁺ at R = 2.0 bohr, −0.6026342144949 Ha to 1e-13), the
  cross-check on `cdft.reference.two_centre`. Full entries: `00_LITERATURE_SURVEY.md`.

### `figures.py` — publication figures of one record (D-72)

`python figures.py <scenario> [--bond-length R] [--xc F] [--n-electrons N] [--gates quick|full]`
solves through `solve_scenario`, attaches the gate report, writes the record beside the figures and
draws; `--from-corpus <file>` draws a stored record without solving. Nothing in `src/cdft` imports
it, and matplotlib (the `figures` extra) is imported only when a figure is drawn. House style and
file list are D-72's.

- **The density is evaluated through the factor**: interpolating `n` cuts the cusp and rings beside
  it, so the figures carry `ρ = n/f²` and the analytic factor as the solver does, with
  `∇n = f²(∇ρ − 2ρ∇u)` and `H_n = f²[H_ρ − 2(∇ρ∇uᵀ + ∇u∇ρᵀ) − 2ρH_u + 4ρ∇u∇uᵀ]`; tests pin the
  evaluator to 1e-10 against the exact hydrogen density and 1e-8 against autograd on a closed form. A
  potential's halo is unknown, so its stencil is shifted inward (a zero halo puts a jump four cells
  deep).
- **A Kato number in a figure is not the solver's cusp**: `d ln n` jumps by exactly `4Z` across a
  nucleus because of the factor, at any spacing — construction, not measurement — and the
  interpolant of `ρ` adds its own slope kink at the node, reported separately. G1.11 measures the
  cusp, on the grid values.

### The namespace placeholders

`cdft/diagnostics/`, `cdft/inversion/`, `cdft/observables/`, `cdft/pseudo/` and `cdft/td/` hold only
an `__init__.py`. They are deliberate reservations for named future increments, not abandoned code:
`diagnostics` for the self-interaction diagnostics, measured and recorded but never thresholded
(D-20); `inversion` for Kohn–Sham inversion, core rather than optional (D-17), exact in closed form
for one- and two-electron systems and Wu–Yang or ZMP for the rest; `observables` for energies, forces
and analysis of a converged solution (I7); `pseudo` for ONCV pseudopotentials in Kleinman–Bylander
separable form (I5); `td` for real-time TDDFT propagators (I11), reachable without changes below the
Hamiltonian seam (D-06).
