# 05 — Decision log

One entry per decision, in numerical order: what was decided, the measured reason, the status. An
amended decision names the amending id. Amending `contract.py` or a goal in `01_PROJECT.md` Part B
requires a new entry here and a `CONTRACT_VERSION` bump.

## Contract versions

| version | date | decisions | what changed |
|---|---|---|---|
| 1.0.0 | 2026-09-12 | D-15 | the frozen API; amendments need a log entry and a version bump |
| 1.1.0 | 2026-09-12 | D-19, D-20, D-21 | non-local density features; diagnostics; scans |
| 1.2.0 | 2026-09-12 | D-27–D-31 | external potentials and the all-electron path; `ElectronSpec`; `ReferenceValue`; the component-gate, diagnostic and scan protocols; `measurements`; inversion; `fingerprint()` |
| 1.3.0 | 2026-09-12 | D-35 | `ExternalPotentialSpec.cusp_factorisation`, default True |
| 1.4.0 | 2026-09-13 | D-37, D-42 | `GateVerdict.DEFERRED` / `RunStatus.DEFERRED`; derived-grid fields |
| 1.5.0 | 2026-09-14 | D-49 | `ComponentGateProtocol` defined at last; `SCFStepProtocol` |
| 1.6.0 | 2026-09-15 | D-60 | `Provenance` device block: six strings plus `gpu_peak_bytes` |
| 1.6.1 | 2026-09-18 | D-80 | `InversionProtocol` loses the `gate_id` / `evaluate(numerics)` pair copied from `ComponentGateProtocol`; no record field changes, older records stay readable |

## Decisions

### D-01 — The theory is ground-state Kohn–Sham DFT

**Decision.** Ground-state KS-DFT, not TDDFT, orbital-free DFT or classical (Evans) fluid DFT;
"classical" in the project name means non-neural.
**Why.** It is the only rung on which learned functionals have reached real chemistry — DM21 [C1],
Skala [C2] and aPBE0-ML, the "fully machine-learned" class of [A17], all replace E_xc in a
conventional SCF.
**Status.** ACCEPTED; D-06 is the seam if the work ever moves to quantum dynamics.

### D-02 — Systems are isolated 3D molecules, non-periodic

**Decision.** No k-points, Brillouin-zone integration, band structures or symmetry reduction, and no
solid-state benchmark including the Δ-gauge protocol of [B4].
**Why.** The corpus target is molecular, and reversal is expensive: k-points touch the eigensolver,
the mixer and the schema.
**Status.** ACCEPTED.

### D-03 — Discretisation is a real-space uniform grid with high-order finite differences

**Decision.** A uniform grid with high-order stencils, over a Gaussian basis or plane waves.
**Why.** No basis set means no basis-set superposition error contaminating energy *differences*, the
only comparable quantity [B4], [B5]; densities are already the grid tensors an XC network consumes;
and h is the one convergence parameter, with a published theory [D1], [D2], [D6].
**Status.** ACCEPTED. Totals are pseudopotential-level and vacuum padding is wasted (D-08).

### D-04 — Mixed precision with float64 gates

**Decision.** float32 hot path; float64 unconditionally for the Gram matrix, orthonormalisation,
Rayleigh–Ritz, energy accumulation, the convergence test and the mixing history, with every scenario
re-run in full float64 on CPU (G2.10) and a disagreement failing the run.
**Why.** On the GTX 1660 Ti (6 GB, float64 at 1/32 of float32) a full-cube float64 orbital block for
100 atoms needs **9.9 GB against 6 GB of VRAM** — feasibility, not optimisation.
**Status.** ACCEPTED; the hot path in fact runs float64 on every device (D-60), and G2.10 belongs to
the float32 hot path (D-74).

### D-05 — libxc is a dependency, but not on the hot path

**Decision.** PW92, VWN5, PZ81, PBE, B88/LYP and r2SCAN are native in torch and differentiable;
libxc [I1] is a pointwise oracle (G0.6, 1e-10 on 10⁵ random points) and PySCF [I2] an
independent-implementation oracle.
**Why.** libxc is **CPU-only**, so a GPU hot path calling it round-trips per SCF iteration, and a
learned functional needs `torch.autograd.grad` *inside* its own `evaluate` to produce v_xc at all.
**Status.** ACCEPTED; native functionals carry a transcription risk bounded at 1.4e-11 by G0.6.

### D-06 — Architect for TDDFT, build the ground state

**Decision.** `HamiltonianProtocol.apply` is the single seam between the SCF, the eigensolver and any
future propagator; H is never materialised as a matrix.
**Why.** Every propagator in [H2], [H3] needs only H·Ψ, so the cost is ~0 — matrix-free is what the
Chebyshev filter [F1] and the memory budget want anyway — and Increment 11 becomes additive.
**Status.** ACCEPTED.

### D-07 — Target scale: 50–100 atoms for single runs

**Decision.** Size the eigensolver (Chebyshev filtering, not direct diagonalisation), the mixer
history and the memory model for 50–100 atoms.
**Why.** The tension with the corpus goal, which wants many small runs, is resolved by sharing one
core and differing only in I9's execution strategy.
**Status.** ACCEPTED.

### D-08 — The computational domain is masked, not a cube

**Decision.** Restrict the grid to the union of spheres around the atoms, as PARSEC [D3] and Octopus
[D5] do.
**Why.** It removes ~55 % of the enclosing cube's points for a typical organic molecule — 100 atoms
from 4.95 GB to 1.77 GB in float32 — at zero accuracy cost, the removed points being vacuum below
G3.2's boundary tolerance.
**Status.** ACCEPTED.

### D-09 — ONCV norm-conserving pseudopotentials only, in v1

**Decision.** SG15 [E2] and PseudoDojo [E3] in Kleinman–Bylander form [E4]; PAW [E6] and ultrasoft
[E7] deferred indefinitely.
**Why.** GPAW cross-checks (G4.3) are then PAW-based, so only *differences* compare there; Octopus
[D5], sharing this project's discretisation and pseudopotential family, is the primary oracle.
**Status.** ACCEPTED.

### D-10 — Spin polarisation deferred (originally to Increment 8)

**Decision.** ONCV alone is the v1 floor, but the spin index exists from I1 with extent 1: every
density, potential and orbital is `(n_spin, ...)` and every XC call takes `(n_up, n_down)`.
**Why.** Open-shell free atoms are spin-polarised, so atomisation energies and the W4-11 target [B2]
are unreachable without spin; the mitigation makes the later change a configuration value plus LSDA
branches, not a reshape.
**Status.** The flagged open risk is resolved by **D-51**, which promotes spin to Increment 4.

### D-11 — Occupations are floating point from day one

**Decision.** Fractional occupations are expressible in the schema from I1; independent of D-10.
**Why.** DM21 [C1] was trained with explicit fractional-charge and fractional-spin constraints and
[A17] reports Skala's largest error on the self-interaction subset SIE4x4 at 13.6 kcal/mol, so the
fractional-electron region is the corpus's most valuable part. Cost: zero.
**Status.** ACCEPTED.

### D-12 — Reproducing known failures is a gate, not a bug

**Decision.** G4.5 requires stretched H₂⁺ to show the textbook semi-local self-interaction error
[A9], [C1]; a solver that gets H₂⁺ *right* at LDA has a bug in its Hartree term.
**Why.** A sharply characterised published failure validates the machinery as strongly as a success,
and where functionals fail is where a learned one has something to learn.
**Status.** ACCEPTED; restated as a sign test by D-55, the magnitude having no closed form.

### D-13 — Gates run at solve time and travel with the data

**Decision.** Gates are evaluated during every solve and their values written into every record; a
record failing any gate is `INVALID` and excluded by the default corpus iterator.
**Why.** The failure mode this project is built against is the silent one — a result that is
unphysical yet looks publishable — and a number without a verdict is exactly that.
**Status.** ACCEPTED; no code path in `cdft` emits a number without a verdict.

### D-14 — The corpus stores fields, not summaries

**Decision.** Converged densities, gradients, τ, all potentials and the full SCF trajectory are
stored on the grid; τ is stored even for LDA runs.
**Why.** [B7] shows functional error and density error cannot be separated from energies alone, so an
energy-only corpus teaches the sum of two unrelated things; τ costs one array and is the rung-3
ingredient [A3] a Skala-like functional consumes [C2].
**Status.** ACCEPTED.

### D-15 — `contract.py` is the frozen API

**Decision.** `contract.py` is the frozen API: implementations are written against it, and it changes
only with an entry in this log and a `CONTRACT_VERSION` bump stamped into every record.
**Why.** A contract that implementations may edit is not a contract.
**Status.** ACCEPTED. Contract 1.0.0.

### D-16 — ML has exactly three roles here, and they are not interchangeable

**Decision.** ML can (i) **replace the unknown** — learn E_xc or v_xc, the only role in which it can
beat what trained it; (ii) **speed up the known**, reaching the same fixed point; (iii) **replace the
solver** with a surrogate, capped by its training data.
**Why.** KS-DFT is exact in principle, so E_xc is the only genuine unknown: ML that changes only the
*path* is verified by the solver itself (G3.4, initial-guess independence to 1e-7 Ha), while ML that
changes the *fixed point* defines the physics and is checked only by the gate suite.
**Status.** ACCEPTED; in `contract.py` as `InitialGuessProtocol` and `XCFunctionalProtocol`.

### D-17 — Kohn–Sham inversion promoted from optional (I10) to core

**Decision.** Inversion becomes core in Phase 2, immediately after the molecular benchmarks, and O-4
is resolved toward the **potential**: the downstream learning target is v_xc, not E_xc.
**Why.** [C19] trains neural LDA and GGA functionals on exact XC potentials from inverse DFT on CI
densities, using **five atoms and two molecules**, and reaches SCAN-level accuracy on hundreds of
molecules outside the training set. Every Phase-1 system has one occupied spatial orbital, so
`EXACT_ONE_ORBITAL` inversion is exact to discretisation error.
**Status.** ACCEPTED; adds `InversionKind`, `InversionSpec`, `InversionProtocol` and the inverted
potentials (D-31).

### D-18 — Do not label the work "PINNs"

**Decision.** Externally: *physics-constrained*, *differentiable* or *constraint-satisfying* learned
functionals. Internal naming unaffected.
**Why.** Strict PINNs fit the KS eigenproblem poorly and carry a live credibility problem [J3]–[J5];
the physics-informed idea that works here is [C3], the KS equations as regulariser.
**Status.** ACCEPTED for external communication.

### D-19 — Non-local density features admitted to the contract

**Decision.** `NonlocalFeatureProtocol`, `NLDFSpec`, `NonlocalFeatureKind`, `XCOutput.v_nldf` and an
`nldf` argument on `evaluate`; every v1 functional stays semi-local and passes `None`.
**Why.** One-electron self-interaction is not removable by semi-local ingredients and is the
project's target (D-12); of the routes past that limit, non-local features of the density [C26],
[C27] are affordable — **quasi-linear O(N log N) on a uniform grid via FFT convolutions** against
O(N⁴) for exact exchange.
**Status.** ACCEPTED. Contract 1.0.0 → 1.1.0.

### D-20 — Diagnostics are a category distinct from gates

**Decision.** `Diagnostic`, `DiagnosticReport`, a `diagnostics` group in `CORPUS_LAYOUT` and
`RunArtifact.diagnostics`: measured, recorded, carried with every record, **never thresholded**.
**Why.** A gate asserts the *solver* is correct; a diagnostic measures how a *functional* behaves,
and here the interesting functionals behave badly on purpose — G4.5's magnitude has no correct value
to threshold, being what a learned functional will be trained to reduce.
**Status.** ACCEPTED. Contract 1.1.0; suite in `01_PROJECT.md` Part C.

### D-21 — Scans are a first-class record type

**Decision.** `ScanKind`, `ScanSpec`, `ScanArtifact`, `SCAN_LAYOUT`; a scan carries its
`integer_endpoints` and its curve-level diagnostics.
**Why.** Self-interaction is a property of curves — the convexity of E(N) [A21], the non-flatness of
E(N, M) [A23] — which single-point records cannot express, and reconstructing curves afterwards
severs the link to the runs that formed them.
**Status.** ACCEPTED. Contract 1.1.0.

### D-22 — The corpus is reframed around fractional-occupation scans

**Decision.** For Phase 1 the corpus is few systems scanned densely over fractional charge and spin
with analytically exact references — the **1 188-run** catalogue — not many molecules at one
functional; success criterion S3 is amended for Phase 1.
**Why.** By [A21] the exact energy at fractional N is the straight line between integer endpoints and
by [A23] the exact energy at fractional spin is constant, so **no correlated calculation is ever
needed at fractional occupation**; the catalogue is ~**6.6 GPU-hours**, against Skala-scale compute
and CCSD(T) labels for the general-chemistry route.
**Status.** ACCEPTED for Phase 1; partially answers O-5, and O-3 depends on I9.

### D-23 — First-class all-electron / model-potential path, with a point nucleus

**Decision.** Bare −Z/r potentials and soft-Coulomb model variants are a supported mode, with a
**point** nucleus regularised by the cell average of 1/r; the smeared nucleus was declined.
**Why.** The cleanest self-interaction systems are one- and two-electron all-electron systems, where
a pseudopotential removes the core the error is defined against, and published all-electron
Gaussian-basis energies use point nuclei.
**Status.** ACCEPTED as `ExternalPotentialKind.NUCLEAR_COULOMB` (D-27); its cost was measured in
D-32 and paid off by D-35.

### D-24 — The repository contains no machine learning, and this is enforced

**Decision.** Exactly three things cross the boundary — the corpus, `XCFunctionalProtocol`,
`NonlocalFeatureProtocol`; `xc/external.py` only loads and evaluates an outside functional, and G5.7
enforces the boundary by static import scan in CI.
**Why.** A combined solver-and-learner cannot be debugged: a wrong number could come from the physics
or the fit and no later experiment separates them. Differentiability is a property of the instrument,
not of learning ([C3], [C4]).
**Status.** ACCEPTED; G5.7 also scans the root modules (D-57) and `figures.py` (D-72).

### D-25 — Two phases: reach a result before finishing the instrument

**Decision.** **Phase 1 (~27 days)** is Track A — grid, XC layer, SCF, all-electron path, spin, SIE
diagnostics — scoped to Z ≤ 2, N ≤ 2; **Phase 2 (~33 days)** is the rest, with inversion core (D-17).
**Why.** The increments sum to ~59 days against a one-month budget, and the cleanest SIE systems need
neither pseudopotentials, GPU, forces nor the benchmark suite; Track A ends in a publishable result,
the diagnostic suite for LDA, PBE and r2SCAN on H, He⁺, H₂⁺ and H₂.
**Status.** ACCEPTED; supersedes O-1 and O-6 as ordering questions; its Z ≤ 2 scope is superseded by
D-47.

### D-26 — Validate the corpus with one throwaway learning experiment, outside this repository

**Decision.** Once Track A produces its first scans, fit a two- or three-parameter GGA enhancement
factor to the project's own PBE data **in a scratch repository, never in `cdft`**, and check that it
recovers PBE's parameters.
**Why.** A corpus can be converged, gated and reproducible and still contain no learnable signal, and
nobody finds out until someone trains on it.
**Status.** ACCEPTED as a recommendation to the downstream project; it does not violate D-24, being a
different repository and a pass/fail on the *data*.

### D-27 — External-potential specification; `pseudo` becomes optional

**Decision.** `ExternalPotentialKind`, `ExternalPotentialSpec` and `ScenarioSpec.external` /
`.all_electron`; a scenario whose potential and pseudopotential table disagree is refused either way.
**Why.** The only expressible external potential had been a sum of pseudopotentials, so the Phase-1
systems could not be written down; a silent bare-Coulomb substitution would label an all-electron run
as a pseudopotential one, so `PSEUDOPOTENTIAL` raises until I5.
**Status.** ACCEPTED. Contract 1.1.0 → 1.2.0; implements D-23.

### D-28 — `ElectronSpec`: fractional electron number and fractional magnetisation

**Decision.** `n_electrons` (float or derived), `magnetisation`, `spin_polarised`,
`fixed_occupations`; the effective `n_spin` moves onto it.
**Why.** A charge and an integer multiplicity cannot express the flat plane's second axis, and by
[A23] exact E(N, M) is flat in M, so measuring the deviation needs a non-integer M.
**Status.** ACCEPTED. Contract 1.2.0.

### D-29 — `ReferenceValue`: a published number without its method is not a reference

**Decision.** Value, units, kind (analytic / literature / cross-code), method, citation and
tolerance, **refusing at construction** to be a literature value without a method.
**Why.** A basis-set-limit energy and a double-zeta energy are not trustworthy to the same digit, so
the tolerance belongs to the number; an untranscribed value is *absent* and its gate reports SKIPPED.
**Status.** ACCEPTED. Contract 1.2.0; `physics_config.py` carries analytic references only.

### D-30 — Protocols for component gates, diagnostics and scans; `measurements` on runs

**Decision.** `ComponentGateProtocol`, `DiagnosticProtocol`, `ScanDiagnosticProtocol`,
`ScanRunnerProtocol`, `RunArtifact.measurements`.
**Why.** G0.1, G0.2, G0.6, G1.1 and G1.9 verify an *operator at a configuration*, not a run, and
`GateProtocol` would force them to fabricate a `RunArtifact`; `measurements` holds readings a solve
cannot recover afterwards, including the grid overrides of D-33.
**Status.** ACCEPTED; `ComponentGateProtocol` was exported from 1.2.0 but not defined until D-49.

### D-31 — Inversion enters the contract; `fingerprint()` implemented in it

**Decision.** `InversionKind`, `InversionSpec`, `InversionProtocol`, the two inverted potentials in
`CORPUS_LAYOUT`, and `fingerprint()` as the SHA-256 of the canonical sorted-key JSON.
**Why.** An unregularised iterative inversion is refused: it admits grid-scale oscillations that
reproduce the target density exactly [G4] — success in every diagnostic except the potential itself.
**Status.** ACCEPTED. Contract 1.2.0; implements D-17.

### D-32 — The point-nucleus path does not reach chemical accuracy by refinement

**Decision.** Recorded with five options, recommending **C′ — factor the cusp out, ψ = exp(−Z r)·φ**
over accepting the limit, a smeared nucleus, adaptive grids and pseudopotentials.
**Why.** No polynomial stencil represents a cusp, so its order is irrelevant: the fitted exponent is
**q = 1.05 (H), 0.62 (He⁺)** against a nominal 8, hydrogen errs by **1.75e-3 Ha at h = 0.12 bohr**
and helium by **1.51e-2 at h = 0.10**, and helium would need h ≈ 1e-3 (~10¹² points). The transformed
operator's probe converges at order **8.2 (H)** and **7.4 (He⁺)**; a smeared nucleus would shift the
energy by ΔE ≈ 2 Z⁴ σ² (+320 mHa for He at σ = 0.1 bohr), voiding published point-nucleus values.
**Status.** SUPERSEDED by D-35 (C′ implemented, O-8 closed) and D-50 (G1.10 retired); the
extrapolation was never gated on, a finer rung having moved it 9.2e-4 → 1.8e-3.

### D-33 — Numerics fixed by the physics are derived, not configured

**Decision.** `grid_config_for_scenario()` derives them in one place and **returns every override so
it is written into the record** (G5.5); the origin is snapped onto a grid point and every run records
`nucleus_offgrid_bohr` per atom.
**Why.** Two I1 counterexamples converged beautifully to the wrong problem: an infinite well of edge
L on a 16-bohr box returns the 16-bohr spectrum (relative error **0.63**, residual 8.7e-7, all other
gates green), and a hard wall needs odd reflection of the halo where vacuum needs zeros.
**Status.** ACCEPTED; fallbacks are allowed, hidden fallbacks are not. Extended by D-42, D-53, D-64.

### D-34 — Gate-suite review after Increment 1: two gates added, thresholds tightened

**Decision.** Added **G0.9** (gradient convergence order) and **G1.11** (Kato cusp); tightened G0.3
orthonormality 1e-10 → **1e-12** and G1.5 charge normalisation 1e-8 → **1e-12**.
**Why.** Nothing gated the gradient, which every GGA and meta-GGA consumes as |∇n|² (measured order
**7.867**), and G1.11 finds from one solve what G1.10 needed a five-point ladder for — it first
caught the off-grid H₂⁺ nuclei at −1.7964 against −2.0. A threshold thousands of times above the
measurement cannot detect a regression: G0.3 measures 1.3e-15 … 1.8e-15, G1.5 2.2e-16 … 8.9e-16.
**Status.** ACCEPTED.

### D-35 — Cusp factorisation implemented and enabled by default

**Decision.** Solve ψ = exp(−Σ_a Z_a|r − R_a|)·φ, leaving `A φ = −½∇²φ + (∇u)·∇φ − ½|∇u|²φ` with no
singular term. On by default: with no nuclei the factor is 1 and `apply` is bit-for-bit unchanged.
**Why.** Hydrogen 1.05e-2 → **2.35e-6 Ha** and He⁺ 7.77e-2 → **2.52e-9 Ha** on the same grid and
solver; D-32's O(h¹) is gone and the error is set by the box. The collocation form is unusable
(weighted self-adjointness 5.6e-3 (H), 1.0e-1 (He⁺)), and zeroing the undefined unit vector at a
nucleus lost Z_a (T = 0.48986 against 0.5).
**Status.** ACCEPTED; closes D-32 and O-8; AMENDED by D-38, D-39, D-54. Contract 1.2.0 → 1.3.0, with
`Measure`, `CuspFactor` and **G0.12** (1e-17); G0.4 is SKIPPED on this path.

### D-36 — An independent radial reference solver

**Decision.** `cdft.reference.radial` solves the separated one-dimensional problem on a
**logarithmic** grid by **direct banded diagonalisation**; G4.8 thresholds it at chemical accuracy.
**Why.** This oracle shares nothing with the 3-D grid, stencil, halo, cusp transform or eigensolvers,
and resolves the cusp with ~1400 points inside one bohr. Validated before use to **4.5e-12** on
hydrogenic levels at Z = 1 and 8.4e-10 at Z = 8, ~1 s per solve; against the 3-D solver at h = 0.25,
hydrogen **2.03e-5 Ha** and He⁺ **6.73e-11 Ha**. Accuracy is set by `r_min`, not the point count.
**Status.** ACCEPTED; a static import scan asserts it never depends on the code it checks. Its
density's trust radius is O-25 (D-72).

### D-37 — DEFERRED is a first-class verdict

**Decision.** `GateVerdict.DEFERRED` / `RunStatus.DEFERRED`, precedence `FAIL > DEFERRED > MARGINAL
> VALID`, legal only with a **named owning increment** in `cdft.gates.catalogue`;
`contract.TRUSTED_STATUSES = {VALID, MARGINAL}` is the single definition behind G5.4.
**Why.** An unevaluable gate used to be a plain FAIL, identical in the record to broken physics: the
suite showed seven red rows of which five were expected. A gate in neither the implemented set nor
the deferral list is still a hard FAIL, so forgetting twice cannot launder a missing gate into a
deferral.
**Status.** ACCEPTED. Contract 1.4.0; deferred records are excluded from the corpus like INVALID.

### D-38 — The cusp-factorised operator is discretised in staggered divergence form

**Decision.** `A φ = (1 / 2 f²) Σ_d D_dᵀ [ f²_face,d D_d φ ] + W φ` — self-adjoint by construction at
any order for any weight, no null space; `energy_matrix` is removed, leaving one operator.
**Why.** Collocation measured relative asymmetry 4.9e-2 (H), 3.1e-1 (He⁺), 4.0e-2 (H₂⁺), and a
Chebyshev filter on a non-symmetric operator does not converge. After: asymmetry 2.3e-16 … 3.2e-16,
checkerboard quotient +25.7 Ha, `h2plus_R2` converged in 6 iterations at 1.5e-9. The symmetric
Galerkin alternative carried a spurious state at −Z²/2, **exactly the true ground-state energy**.
**Status.** ACCEPTED; supersedes the `energy_matrix` half of D-35 and returns G2.4 to thresholding
the residual.

### D-39 — The transformed potential is cell-averaged near each nucleus

**Decision.** Replace `W` and `∇u` by their cell averages within two spacings of a nucleus, by a
4-point Gauss–Legendre product rule; the on-nucleus `lost_square` correction is deleted.
**Why.** `W = −|∇u|²/2` holds one unit vector per nucleus, discontinuous *at* it: H₂⁺ at R = 2
measured 2.52e-3 (h = 0.50, on grid), 3.74e-3 (0.40), **1.13e-2 (0.32)**, 1.52e-4 (0.25, on grid) —
a finer grid giving a worse answer. After: 1.04e-2, 3.60e-3, 1.94e-3, 1.02e-3, monotone, order 3.3.
Radius 2 beat radius 1, which gives smaller numbers but is not monotone.
**Status.** SUPERSEDED by D-54 (`W` goes through the lumped quadrature). The residual multi-centre
constant, 1.0e-3 Ha at h = 0.25, was O-13.

### D-40 — The gate catalogue is the single source of truth for a gate

**Decision.** `cdft.gates.catalogue` holds every gate with a `Lifecycle` of `IMPLEMENTED`,
`DEFERRED` (with the increment that will deliver it) or `OUT_OF_PHASE_1` (with phase); thresholds
stay on the implementation's `GateSpec`.
**Why.** The gate set lived in three places that could disagree silently: the audit found a deferral
table misattributing G0.7 and G1.7, G1.11 implemented and load-bearing while absent from the
specification table, and twelve gates claimed implemented with no evaluator. At the decision:
**54 gates, 24 implemented, 25 deferred, 5 out of Phase 1** (56 / 46 after D-70).
**Status.** ACCEPTED.

### D-41 — A second oracle: H₂⁺ by prolate-spheroidal separation

**Decision.** `cdft.reference.two_centre` solves the separated spheroidal equations and
Richardson-extrapolates over three nested grids, **refusing** the extrapolation when the re-measured
convergence order is outside 10 % of 2.
**Why.** The radial oracle covers spherical systems only, leaving the multi-centre path on a single
seven-digit table number; this agrees with Madsen and Peek (1971) to **2.2e-9 Ha at R = 2, in 0.2 s**
at any bond length, and it is what found D-39. Two traps: ξ = 1 and η = ±1 are regular singular
points, and a vertex-centred grid converged to −0.2588 instead of −0.6026.
**Status.** ACCEPTED.

### D-42 — The all-electron grid is derived from the structure, not configured

**Decision.** `coulomb_box_edge` and `coulomb_spacing` compute both values from the structure, every
choice recorded as an override (G5.5): **single centre** — edge `32 / Z` bohr, `POINTS_PER_EDGE = 33`
always; **multi centre** — edge `nuclear span + 12` bohr, spacing **0.25** bohr. The one explicit
bypass is `derive_grid=False`, for the ladder gates G3.1 and G3.2.
**Why.** `ALL_ELECTRON_NUMERICS` was `h = 0.125` in a 12-bohr box, tuned for the *bare* path and
never re-tuned after D-35 inverted the requirement: hydrogen at 97³ points is **1.7e-04 Ha in 172 s**
against derived (`h = 1.0`, box 32) 33³, **1.9e-13 Ha in 0.7 s**. The spacing rule is a *point
count*: at 11 points per edge He⁺ loses eight orders and G1.11 degrades from 3.2e-10 to 9.5e-04. A
single centre is **box-limited**, reaching round-off at `16 / Z` (`COULOMB_HALF_BOX_PER_Z`), and
refining `h` at fixed box *hurts* (4.7e-14 → 1.2e-12, 0.28 s → 262 s), whereas H₂⁺ measures 1.04e-02
Ha at every edge from 10 to 24 bohr.
**Status.** ACCEPTED; supersedes the grid half of `ALL_ELECTRON_NUMERICS`, named presets being
rejected because nothing stops a scenario running at the other regime's settings. AMENDED by D-48
(the molecular 0.25 is a *cost* choice: `h = 0.125` takes H₂⁺ to **8.1e-05 Ha** for ~30× the CPU
time), D-53 and D-64; **H₂ at R = 1.4 stays 2.0e-03 Ha, above the 1.594e-03 Ha chemical-accuracy
bar** (`02_STATUS.md` Part B, A-1).

### D-43 — A known failure keeps its verdict and loses its power to fail the build

**Decision.** A known-open failure keeps its FAIL verdict and its record stays INVALID, losing only
the power to set the exit code; entries are keyed to **one gate on one scenario** and expire both
**upward** (past `recorded × drift`) and **downward** (reported as stale).
**Why.** G2.7 and G3.1 produced four failing molecular gates and every failure was *correct* (O-13).
Raising thresholds is worst — G2.7's 1 meV/atom is what forces must meet for geometry — deleting or
skipping hides the failure, and a permanently red suite disarms G5.6's exit code.
**Status.** ACCEPTED; each entry names an open item, its value is pinned in
`tests/golden/values.json`, and six unit tests cover the mechanism.

### D-44 — The weak convergence criterion applies only on the weak-form path

**Decision.** The Ritz-drift stop criterion of `ChebyshevFilteredSubspace` is gated on
`uses_divergence_form`; the stop reason is recorded as `eigen_stop_reason` and checked by G5.5.
**Why.** It exists for a real reason (D-35) — on the cusp path He⁺ is right to 8e-11 Ha with a
collocation residual of 1e-2 — but on a plain collocation operator it stopped the harmonic well at
**2.6e-6** and the box at **1.0e-5**, both against `residual_tol` 1e-8 and both reported
`converged`. After: the box reaches **8.6e-9** in 44 iterations and G2.4 passes on both model systems.
**Status.** ACCEPTED; AMENDED by D-55.

### D-45 — One implementation of "does this report fail the build"

**Decision.** `known_open.partition_failures` splits failures into **new** (a defect, or a known one
past its drift allowance — sets the exit code), **known** and **unimplemented**; `test_suite.py` and
`cdft.run` both delegate to it.
**Why.** After D-43 the two callers classified differently, so `python -m cdft.run h_atom` printed
**FAILED** on a run whose every gate passed (19 pass, 0 fail), purely because G1.10 is a registered
known failure.
**Status.** ACCEPTED.

### D-46 — Inside the autograd path, torch; outside it, whatever is fastest

**Decision.** Inside the `Hamiltonian.apply` seam — the functional's `evaluate`, the Hamiltonian
application, the SCF loop, the density accumulation, every energy integral — is torch without
exception; outside it — oracles, corpus writer, scans, figures, `scripts/` — is free choice and runs
**on the CPU**, not on the card.
**Why.** "PINN-training accessible" means differentiable end to end, much stronger than "runs on a
GPU" ([C3], [C4]), and two CUDA libraries in one process hold two memory pools neither can see,
which makes a 6 GB budget meaningless; CuPy (no autograd) is forbidden inside the seam. torch
2.14.0+cu130's arch list contains **`sm_75`**, the GTX 1660 Ti, so the ordinary wheel works.
**Status.** ACCEPTED. Amended for the device only by D-65.

### D-47 — Scope settled: exact KS solutions for molecules; training is a separate project

**Decision.** Autograd *inside the functional* is required and already provided (G0.5 gates it);
autograd *through the SCF* is out of scope, but I3 keeps its one-step fixed-point map **extractable
from the loop**. "Exact" means exact against the Kohn–Sham equations, not against experiment or
CCSD(T).
**Why.** The unrolled route costs 7.9 GB for water and 559 GB for a 100-atom system against 6 GB, so
if ever wanted it will be implicit differentiation. A solver whose discretisation error is comparable
to the functional error cannot measure that error.
**Status.** ACCEPTED; resolves R1 and R2, R2 being withdrawn as a defect; supersedes the Z ≤ 2 scope
of D-25. C2 was identified here as blocking and C1 demoted, both closed by D-54 and D-58.

### D-48 — A power law fitted to the coarse end and evaluated at zero is not a limit

**Decision.** **G3.1 now refines** — three rungs at ratio 1.25 downward from the production spacing,
the verdict being the difference between the two finest rungs; the extrapolation is reported beside
it and never thresholded. ~130 s per molecule, `full` profile only.
**Why.** Fitted against H₂⁺ at `h ≥ 0.25` the coarsening ladder returned order 7.15 and an `h → 0`
limit of −1.103511 Ha, 8.8e-4 from the oracle, published in four documents as a physical finding.
Refining disproves it: 1.02e-3 (h = 0.25), 4.32e-4 (0.20), 1.58e-4 (0.16), **8.09e-5 (0.125)** — an
order *below* the limit declared unreachable.
**Status.** ACCEPTED; re-characterises O-13 as a resolution problem.

### D-49 — Contract 1.4.0 → 1.5.0: `ComponentGateProtocol` and `SCFStepProtocol`

**Decision.** `ComponentGateProtocol` is defined with the signature `cdft.gates.base.ComponentGate`
already implements; `SCFStepProtocol` — pure, free of mixer state, no in-place mutation — moves the
surviving requirement of D-47 into the contract.
**Why.** `__all__` had exported `ComponentGateProtocol` since 1.2.0 and D-30 recorded it as added,
but it was never defined, so `from contract import *` raised. Also corrected: docstrings still
asserting the pre-D-47 "must backpropagate through the SCF", and a version note listing 1.3.0 before
1.2.0 and omitting 1.4.0.
**Status.** ACCEPTED.

### D-50 — G1.10 retired; G1.13 (cusp-weight quadrature) added

**Decision.** **G1.10 is retired** — it built its own ladder on the un-transformed −Z/r operator,
which nothing has run by default since D-35. **G1.13** compares the grid quadrature of `f²` with its
closed form (checked against two independent quadratures to 1e-16), at 1e-8 relative, EXACT.
**Why.** G1.5 (∫n = N) cannot see the defect: φ is normalised in the *discrete* measure `Σ f²φ²h³`
and `n = f²φ²` uses the same rule, so normalisation holds by construction on any grid. What that
hides is that the plain rule overestimates `∫f²` by **15.4 % at h = 1/Z**, so **the recorded density
is 13.4 % too small at every point** while the eigenvalue sits at 1e-13 (`n(0)/n_exact(0) = 0.866`).
**Status.** ACCEPTED; closes O-17 and backlog C4, opens O-19 and ledger A-9. G1.13 failed on all five
all-electron scenarios, registered known-open with pins 1.54e-1 (H, He⁺, He), 1.0e-3 (H₂⁺) and
6.1e-4 (H₂), retired by D-54; the G1.10 measurement survives as
`scripts/probes/cusp_factorisation_probe.py`.

### D-51 — Spin polarisation promoted to Increment 4; increments renumbered

**Decision.** I8 → **I4** (spin, G4.6); I4 → I5 (ONCV); I5 → I6 (GPU and the precision policy,
G2.10); I6 → I7 (forces, G2.6, G2.8, G6.1); I7 → I8 (benchmarks G4.2–G4.5); I0–I3, I9–I11 unchanged;
entries dated before 2026-09-14 keep their original numbers and are read with this table.
**Why.** Three independent reasons had accumulated: atomisation energies (D-10), half the flat plane,
and the paper's second axis. I2 already writes the spin-polarised forms of every functional, so I4 is
the SCF's `n_spin = 2` leg, the Fermi-level search of [F20] and the LSD comparison against NIST.
**Status.** ACCEPTED; closes O-1, O-6, O-7 and risk R7. The *order* is amended by D-59 and D-63.

### D-52 — Four findings of the 2026-09-14 review that change Increment 3

**Decision.** (1) The interacting problem is spacing-limited on atoms too, so I3 derives a separate
grid rule and uses cusp-aware quadrature for every density integral. (2) O-11 was bookkeeping: the
standard KS energy never needs `∫n·v_ext`. (3) G1.11 must not assert −2Z at GGA, and `v_xc` is
cell-averaged near nuclei like `W`. (4) The plain cusp factor cannot leave the first row.
**Why.** D-42's atomic grid is exact for the transformed eigenproblem and wrong for anything
integrating the density: the Hartree energy of the 1s density (exact `5Z/16`) is **+0.19 Ha at h = 1
(H)** and **+0.37 Ha at h = 0.5 (He)**, falling as O(h⁴) in `Zh`. PBE gives `v_xc ≈ −0.017/r` on a
hydrogenic density, so `Z_eff ≠ Z` at GGA level, and `φ = ψ/f` grows as `e^{(Z−κ)r}` — **1.9e9 for a
2p at Z = 10 by 4 bohr**.
**Status.** ACCEPTED as constraints on I3; items 1–3 delivered by D-53, D-54, D-55. Item 4 is the Li
probe, unmeasured (`04_ROADMAP.md`); nothing beyond Z = 2 runs before it.

### D-53 — The interacting grid rule: `h = 0.25` and a derived vacuum margin

**Decision.** An interacting scenario (`xc.name != "none"`) runs at `INTERACTING_SPACING = 0.25` bohr
with a vacuum margin beyond the nuclear span of **`max(7, ln(10⁹)/(2κ))`**, `κ = min_a (Z_a −
0.30·max(0, N/n_atoms − 1))`, sized so the tail leaves 1e-9 of the charge outside the box: He and
He⁺ take the 7-bohr floor, H, H₂ and H₂⁺ 10.4 bohr. D-42 stays for non-interacting scenarios.
**Why.** D-54's rule already makes every density integral exact at the D-42 atomic spacing (`∫n − N`
to 1e-10 and `E_H − 5Z/16` to 2e-9 at `h = 1/Z`, against 1.5e-1 and 8e-2 for the plain rule), so what
sets the spacing is interpolating the self-consistent density onto the sphere nodes: the LDA
`v_xc ∝ n^{1/3}` puts an `r³` term into φ that the degree-7 stencil resolves only to O(h³). Helium at
LDA against NIST SRD 141 is **2.64e-5 at h = 0.25** and 5.9e-6 at h = 0.20 — a cost choice, not an
accuracy limit.
**Status.** ACCEPTED, amended the same day: the first version fixed the margin at 7 bohr for
everything, a helium number, and hydrogen at LDA in that box came out **1.24e-4 Ha above NIST**
against 2.3e-6 in a 20-bohr box. AMENDED by D-64.

### D-54 — Cusp-aware quadrature: partition of unity, sphere rules, lumped potentials

**Decision.** Everything integrating the density on the cusp path integrates `f²·g` through a
**partition of unity** `p_a(r) = ½ erfc((s_a − r_c)/w)`, `w = 1.3h`, per nucleus; compactly supported
**sphere rules** with `f²` and `W` analytic at the nodes; **degree-7 tensor Lagrange interpolation**
as a sparse CSR operator; **lumped mass weights** which are the eigensolver's measure; **GGA in weak
form**; and a **Hartree split** `n = Σ_a c_a e^{−2Z_a s_a} + n_s` with a closed-form potential for
the cusped part.
**Why.** The plain `h³` sum is O(h⁴) in `Zh` on that integrand (A-9), and an exact cell integral near
the nucleus spliced onto it breaks the Euler–Maclaurin cancellation. The lumped weights give
`∫n = Σ ω φ² = N` *and* integrate the cusp weight exactly (G1.13 to 1e-10 on atoms), and `W̃ = −Z²/2`
exactly for one nucleus, so single-centre eigenvalues do not move; `E_ext` agrees with the
band-energy complement to 1e-15 (A-6). On the molecules the lumped `W` takes H₂⁺ at R = 2 from
1.02e-3 to **2.6e-5 Ha** at the same h = 0.25 and H₂ from 2.01e-3 to **1.4e-4**, so G3.1 passes.
**Status.** ACCEPTED; closes O-19 as a mechanism, ledger A-9, backlog C7 and **C1 as an accuracy
item**, retiring the five G1.13 known-open rows; C2 is closed by D-58.

### D-55 — Gate restatements forced by the interacting path

**Decision, nine items.** **G1.11** fits a quadratic in r to `ln n`, asserting −2Z at LDA and
reporting SKIPPED with the measured slope at GGA; **G4.5** becomes the one textbook *sign*, that
`E(H₂⁺, R ≥ 6)` lies below `E(H)` at the same functional, the magnitude recorded and never
thresholded; **G0.5** thresholds at 1e-4 with a relative step; **G1.8** is an artifact gate on the
integrated `E_xc / E_x^LDA` (`C_LO = 2.273`); **r2SCAN [A11] is DEFERRED**; **G1.13** subtracts the
tail outside the box; **G2.4** keys on `divergence_form`; **G2.3 stays strict** while the SCF runs
four filter steps *without* the Ritz-stability stop; and **G5.2** re-solves through `solve_scenario`.
**Why.** The linear Kato fit reported a 10 % "violation" on self-consistent helium where the
quadratic recovers −2Z to 8e-3, and at GGA the exact cusp has no closed form (He at PBE −4.090
against the bare −4); nor has the self-interaction magnitude, but its sign is never wrong. With four
filter steps and no Ritz stop hydrogen converges in **16** iterations (was 36), fixed points
unchanged to 5e-12.
**Status.** ACCEPTED; implements D-52 item 3. r2SCAN is O-21.

### D-56 — LOBPCG is preconditioned in the similarity frame, `f⁻¹·K·f`

**Decision.** The Teter–Payne–Allan preconditioner [F7] is applied as `f⁻¹ K (f r_φ)`; all three
frames stay selectable, `f` clamped at 1e-150.
**Why.** [F7] models the plain Laplacian, the operator on ψ = fφ, so on the transformed residual it
preconditioned the wrong operator and was harmful: against CheFSI, H₂⁺ at R = 2 took 140 iterations
unpreconditioned, **300 and unconverged** in the transformed frame and **97** in the similarity frame
(H₂ 88 / 179 / 62; He 67 / 97 / 52). Only this frame is symmetric in the measure.
**Status.** ACCEPTED; closes O-15, ledger A-8, backlog C5 and retires the two G2.5 known-open rows.

### D-57 — `config.py`, `physics_config.py` and `run.py` are canonical at the repository root

**Decision.** The three modules a reader edits sit beside `contract.py` at the root; their `src/cdft`
counterparts are shims aliasing the `sys.modules` entry, so every import path keeps working and there
is exactly one `REGISTRY`.
**Why.** They are the configuration surface of the solver, and G5.7's import scan covers the root
modules as well: moving a file out of the package must not move it out of scope.
**Status.** ACCEPTED. Extended by D-72 (`figures.py`) and D-73 (the setups table in `config.py`).

### D-58 — Overlapping spheres split by erf cells: the two-centre quadrature

**Decision.** Spheres keep their full D-54 profile at any bond length, so they overlap and each
contains the other nucleus; the plain-sum share is split by **cell functions** `β_a = P_a / Σ_c P_c`,
`P_a = Π_{b≠a} ½ erfc((s_a − s_b)/(w_c √2))`, `w_c = 1.0 h`, and an overlapping sphere uses an
angular node spacing of **0.7 h**. A single nucleus is untouched.
**Why.** D-54 capped each sphere at the internuclear distance, so at R = 1.4 the taper narrowed to
`w = 0.78 h` and the plain sum aliased it by `exp(−2π²w²/h²) ≈ 6e-6` of the taper's content —
G1.13's 2.3e-5 on H₂ (A-11) — with the aliasing depending on where the grid fell, the egg-box (A-3).
Becke cells, whose transition width is ∝ R, were tried and leak into the lune no rule covers
(2.4e-6).

| scenario | quantity | D-54 | D-58 | threshold |
|---|---|---|---|---|
| H₂ R = 1.4 | G2.7 egg-box (Ha/atom) | 4.45e-5 fail | **2.90e-5 pass** | 3.67e-5 |
| H₂ R = 1.4 | G1.13 `∫f²` | 2.32e-5 | **3.3e-8** | 1e-8 known-open |
| H₂⁺ R = 2 | G1.13 | 8.4e-8 | **2.3e-8** | 1e-8 known-open |

**Status.** ACCEPTED; closes backlog C2 (ledger A-3) and the last O-13 row, reduces C8 700×, and
leaves G1.13 known-open on the molecules at 2–3× the threshold.

### D-59 — Increment 6 (the GPU port) runs next, before spin

**Decision.** The order after I3 becomes **I6 → I4 → Li → I7 or I5 → I8 → I9**, targeting the
Windows/CUDA development host (GTX 1660 Ti, 6 GiB).
**Why.** Track A's physics does not need the card; its *verification* does — a full gate profile is
~5 h on the 2-core Linux CPU host and was killed there twice for memory, and one H₂⁺ solve on its
D-53 box is an hour. The CPU float64 path stays the reference.
**Status.** ACCEPTED; AMENDED by D-63.

### D-60 — Device residency in float64; int32 CSR on the card; `hot_dtype` until the float32 hot path

**Decision.** Every tensor lives on `grid.device` and the physics is **float64 on the card exactly as
on the CPU**; the float32 hot path is later work. The seeded start is drawn on the CPU generator and
moved, `L`/`Lᵀ` carry **int32** CSR indices on CUDA, `hot_dtype` stays float64, and the contract
gains the `Provenance` device block.
**Why.** A CPU generator cannot feed `torch.randn(device="cuda")` and a CUDA Philox generator draws a
different stream from the same seed, so the acceptance would otherwise compare two RNGs; int32
indices cost 0.35 GB per operator instead of 0.7 for the diatomic rule; and nothing consumed
`PrecisionConfig.hot_path`, so provenance would have said `float32` for a float64 run (G5.1).
Measured on the card: GPU-vs-CPU float64 over nine scenarios gives max |Δ| **2.05e-13 Ha on energies
and 3.27e-13 on eigenvalues** against a 1e-10 acceptance.
**Addendum (2026-09-16).** Host reads were *not* the anomaly — the census predicts 0.9 s of host cost
for the 593-s `harmonic_w1` and 24.6 s for the 1878-s `h_atom_lda`, H4 (D-62) being the cause — but
the SCF path was reduced anyway (`he_atom_lda` 3521 → 332 reads; census table in `02_STATUS.md`
Part F), the CPU fingerprint unchanged. **No arithmetic was re-associated:** a fused
`torch.add(a, b, alpha=−c)` differs from `a − c·b` by **1.8e-12 at c = 1700**.
**Status.** ACCEPTED, measured; the acceptance was met and **O-9 closed**; amended for the device
only by D-65.

### D-61 — Determinism on the card: strict algorithms and a fixed cuBLAS workspace

**Decision.** On CUDA `configure_device` sets the workspace variable with `setdefault` and follows
`NumericsConfig.deterministic`: strict when the numerics ask, torch's default otherwise. The mode
that ran (`strict`, `warn_only`, `off`) is in every record (G5.5).
**Why.** S4 and G5.2 promise reproducibility to 1e-9, which a float32 hot path makes meaningless
unless the kernels are deterministic — cuBLAS is not, by default, for some shapes. Measured: strict
costs **~2×** (3.4 s against 1.6 s on one solve) and G5.2 passed on the card within **2.7e-13 Ha**.
**Status.** ACCEPTED, measured, amended 2026-09-16 so the mode follows the numerics; AMENDED by
**D-69**, which confines strict mode to the G5.2 pair.

### D-62 — The wide triangular solve is chunked on CUDA: a cuBLAS `trsm` cliff at ~2¹⁹

**Decision.** `precision.solve_triangular_wide(L, X)`: on CUDA above `CUDA_TRSM_CHUNK = 2¹⁸` columns
the solve runs over column chunks of that width; on the CPU, or below it, the plain call.
**Why.** Every GPU kernel scales linearly with the grid (61–77 GB/s throughout on 57³–96³ in float64)
except `rayleigh_ritz`: 21 ms at 72³ = 373 248 points and **17 165 ms at 81³ = 531 441**. The culprit
is cuBLAS `trsm`: 2.4 ms at n = 373 248 and **11 525 ms at 524 287**. Chunking is **bitwise
identical** at 4.4 ms; after it, `harmonic_w1` 593 → **4.5 s** and `h_atom_lda` 1878 → **43 s**.
**Status.** ACCEPTED, measured; **H4 closed**.

### D-63 — Order of work after the GPU port

**Decision.** Throughput first, on any card (`full` under 30 min, `quick` under 2 min, settings
derived from the card the solver finds); then closing everything that is not a physics limitation,
including the float32 hot path with G2.10; then the gradient and meta-GGA rungs, closing O-21; then
the figure machinery. **O-23 jumps the queue** and runs first. After that: I4 spin, the D-52 item 4
outcome, I10, I7 forces, I8 benchmarks (O-2) and I9's corpus generator.
**Why.** The instrument runs on the card (O-9 closed) but the audit profile takes 66 min against a
30-min target and the inner loop 174 s against 120; a clean instrument must come before new physics,
since the gradient rungs' grid sensitivity (R4) is readable only against a suite with no unexplained
red rows. O-23 jumps first because it changes the molecular grids.
**Status.** ACCEPTED; amends the order of D-59. AMENDED by **D-74**.

### D-64 — The lattice rule: a spacing on which every nucleus is a lattice point

**Decision.** `lattice_spacing` sets `h = d / m` from the smallest non-zero inter-nuclear coordinate
difference `d`, `m = round(d / target)`, requiring every other difference to be a multiple of `h` to
1e-9; the **edge is rounded up** to a multiple of the spacing, never the spacing to the edge; the
lattice is anchored on the first nucleus for a derived grid, while `derive_grid=False` keeps the
fixed lattice the ladder gates and G2.7 sweep against; and gates read the recorded grid.
**Why.** D-53 snapped the *spacing* to the derived box, so `h2_R1.4_lda`/`_pbe` ran at `h = 0.251401`
with the protons **0.054 bohr** off the lattice; a 0.25 lattice cannot hold both protons of H₂ at
R = 1.4 at all, 5.6 cells apart. A nucleus thousandths of a bohr off the lattice split every cubic
shell into near-equal radii and made G1.11 read **229** on `h2plus_R2_lda`.
**Status.** ACCEPTED, CPU-verified, the interacting molecules re-measured on the card; **closes
O-23**, amends D-42 and D-53, and `nucleus_offgrid_bohr` is **0.0 on every Phase-1 scenario**.

### D-65 — Device-specific arithmetic on the CUDA path: fused kernels

**Decision.** On CUDA the solver may sum in a different order from the CPU code provided the CPU
float64 path stays byte-for-byte the previous code and the GPU-vs-CPU fingerprint diff stays ≤ 1e-10
on all nine scenarios. `cdft.operators.fused` offers `compile`, `gemm` (dense derivative matrices, ~5
launches per axis instead of ~18) and `eager`, the tier chosen once per device and recorded (G5.5).
**Why.** This reverses, for the device only, D-60's rejection of the fused `add(alpha=)` (1.8e-12 at
c = 1700). GEMM and eager are the same linear operator, so only the summation order differs: measured
**2.2e-16 relative at 21×23×25** for both tiers and ≤ 1e-13 on random boxes, with `⟨a|K|b⟩` symmetric
to round-off, so D-38's self-adjointness holds.
**Status.** ACCEPTED (CPU legs measured; card legs pending).

### D-66 — The geometry cache: one build per geometry per process

**Decision.** A process-wide, byte-bounded LRU cache keeps what depends on the geometry alone — the
cusp factor, the quadrature, the step's Hartree-split tensors and the Poisson kernel — keyed by
**value, never identity and never the density**: grid, stencil orders, boundary, mask digest, device,
dtype, CSR index dtype, nuclear charges and positions, the class constants and `CACHE_VERSION`.
**Why.** The 6–10 solves one scenario runs in the `full`/`gpu` profiles rebuilt these every time (the
cusp factor alone is ~16 k launches); the audit found no in-place write into a cached tensor, and a
second solve is bit-identical.
**Status.** ACCEPTED on the CPU (bit-identical); card measurement pending. Budget CPU 3 GiB and CUDA
1.5 GiB by default, overwritten per card by D-70; a diatomic bundle is **2.64 GiB**, above the
default, so it is used but not retained.

### D-67 — CUDA graphs for the Chebyshev filter step (device path only)

**Decision.** On CUDA one filter step — the whole degree-`d` recurrence [F2] *including* the kinetic
stencil and diagonal potential — becomes one `torch.cuda.CUDAGraph` for a fixed block, one graph per
solve. The CPU keeps the eager filter byte for byte.
**Why.** The scalars arrive as rows of a device tensor and the potential as a static buffer, and a
product by a 0-d tensor is the same IEEE operation as by a Python float, so the captured body is
**bitwise the eager filter**. Every capture is validated once at 1e-12, and a mismatch disables the
engine with the reason in the record (G5.5).
**Status.** ACCEPTED; card measurement pending.

### D-68 — Lanczos spectral bounds reused across SCF iterations, widened by the Weyl bound

**Decision.** From SCF iteration 2 the filter reuses the previous interval instead of a fresh 12-step
Lanczos estimate, widened by `δ = maxᵢ |v_eff,new(i) − v_eff,old(i)|`. A fresh estimate is taken on a
level-shift or fallback rung, a rising residual, eight iterations, `δ` above 5 % of the width, or a
top Ritz fraction of 0.9.
**Why.** The margin is rigorous: between iterations only the diagonal potential changes, a diagonal
operator's norm in a positive diagonal measure is its largest entry, and by Weyl's inequality no
eigenvalue moves further. Measured on `he_atom_lda`: applies per warm iteration **102 → 90**, the
fixed point moving **1.8e-15 Ha**.
**Status.** ACCEPTED, measured on the CPU; `CDFT_LANCZOS_REUSE=0` restores the previous path
bit-identically.

### D-69 — Strict mode confined to the G5.2 pair; the allocator set at import

**Decision.** The primary solve of every profile runs with deterministic algorithms **off**, `gpu`
included; G5.2 runs two strict solves of its own and thresholds only `|ε_A − ε_B|` at 1e-9, recording
the primary-vs-strict difference without thresholding it. `expandable_segments:True` is set at import
of `cdft.precision`, and device memory is released after each suite scenario and each solving gate.
**Why.** D-61 measured strict mode at ~2×, a tax on every production solve for a property only the
audit needs; the difference against the strict run measures kernel choice, not reproducibility; and
the 1.34 GiB reserved-but-unallocated at the O-22 OOM is what expandable segments address.
**Status.** ACCEPTED, CPU-verified; card cost to be measured; amends D-61.

### D-70 — A card-agnostic device policy, and G5.9, the sync/launch budget gate

**Decision.** `DeviceProfile` is captured once per process per device and `derive_settings()` is a
pure function of it (geometry cache 25 % of VRAM capped at 8 GiB, `trsm` chunk 2¹⁸, graphs only with
≥ 2 GiB free, concurrent solves `clamp(⌊(free − 0.5 GiB)/2 GiB⌋, 1, 4)`); `configure_device` applies
it and **asserts TF32 off on every card**. **G5.9** (EXACT) thresholds Python reads and implicit
syncs per iteration at loop sites, and launches per solve, against budgets = measured + 10 %.
**Why.** Hard-coded 6-GiB settings do not survive a different card. Per-iteration headroom is capped
at 0.5, so one new sync per iteration always fails. Budgets recorded 2026-09-16 (CPU): `h_atom`
7 CheFSI iterations, 9/7 reads and 16/7 implicit syncs per iteration, **34 841 launches**;
`he_scf_probe` 12 SCF iterations, 80/12 reads and **127 429 launches**.
**Status.** ACCEPTED, CPU-verified.

### D-71 — Throughput of the fractional-occupation scan: how it is measured

**Decision.** One geometry with N swept — default `h2plus_R2_lda`, N = 0.2 … 1.0 — each run going
through `solve_scenario` and a gate suite at production numerics, so runs/hour counts
**gate-certified** records; processes and streams are both measured, the higher valid rate winning.
**Why.** Host RAM is a real constraint, not only VRAM: each worker builds its quadrature and `L`/`Lᵀ`
on the host (D-60), `h2plus_R2_lda` reached **3.5 GiB RSS** and a second worker was killed for memory
on an 8 GiB host.
**Status.** ACCEPTED as a design; card numbers pending.

### D-72 — `figures.py`: publication figures of one record, at the repository root

**Decision.** A root-level `figures.py` beside `run.py` (D-57) picks a geometry and a scenario from
the configs, solves, and draws ten figures of one record: `density_map`, `density_profile`,
`radial_density`, `cusp_and_decay` [A29]–[A31], `density_error` [B11], `reduced_gradient` [A32],
[A33], `potentials`, `self_interaction`, `spectrum` and `scf_convergence` (D1.1, D1.6, D1.7).
Nothing under `src/cdft` imports it, G5.7 scans it as a root module, and `scripts/figures_suite.py`
runs it once per registered scenario. Only `contract.TRUSTED_STATUSES` (VALID, MARGINAL) are drawn by
default; other records holding fields need `--allow-invalid` and carry a badge naming the state and
"not for publication"; a record without fields is never drawn.
**Why.** Two numbers the review corrected: a Kato jump in a figure is the *factor's*, exactly `4Z_a`
at any spacing, and the degree-7 interpolant adds a slope kink of its own (+0.0128 bohr⁻¹ for LDA He,
−0.0138 at each H₂⁺ nucleus), and the first version printed their sum as a "measured jump" (8.0128,
3.986); they are reported separately now, and neither measures the solver's cusp — G1.11 does. The
radial oracle's inner trust radius is set by its Dirichlet truncation at `r_min = 1e-12`: for LDA He,
`n e^{2Zr}` is 4e-6 low at 1e-6 bohr and 0.2 % low at 1e-9, and a spline of `ln n` in `ln r` turned
that into `d ln n/dr = +2e6` at 1e-9. Whether the oracle itself should change is **O-25**.
**Status.** ACCEPTED, CPU-verified (95 tests), amended the same day after an independent review.

### D-73 — Consolidation of 2026-09-17: documents, comments, setups, pruning

**Decision.** The six documents are rewritten to one purpose each — state, not story: closed items
become one line, executed plans are deleted, the running work log becomes the improvements catalogue
(`02_STATUS.md` Part F), open items live only in `02_STATUS.md` Part A §7 and the numerical standard
only in Part A §2. Code comments keep one-line summaries with `D-nn` pointers, and working
by-products are listed for deletion in `prune.ps1`. `config.py` gains the setups table — one `Setup`
per scenario with levels `draft`/`standard`/`fine`/`reference`, `standard` being the production rule
and the only level comparable with golden values and the known-open registry, the others running
inside `cdft.grid.grid_rule` and recorded in `grid_overrides["grid_rule"]` — plus
`run.py --setups/--res`.
**As delivered.** `config.numerics_for(scenario_id, level, **overrides)` returns `(numerics,
grid_rule targets)` and `numerics_for_spec` resolves a `--scenario-file` scenario that has no setup.
A model-system level must divide the box edge exactly, or `UniformGrid.from_config` snaps `h` and
G5.5 flags the override — hence per-scenario model levels rather than one global ladder. `reference`
on `h2plus_R8_lda` (181³ = 5.9 M points) and on `h2_R1.4_lda/pbe` (144³ = 3.0 M), and `fine` on
`h2plus_R8_lda` (3.0 M), do not fit a 6 GB card; `run.py` refuses `--res` or a field override
together with an explicit `--numerics` preset or YAML and **exits 2**. The audit's dead code is
removed (−103 lines) and its bit-identical speed actions applied: `UniformGrid.partial_derivative`
where one gradient component was taken of three (4.11 → 0.46 ms per call at 60³), the wasted clone
and second spin reduction in `KohnShamStep.potentials`, memoised golden solves (62 → 35 s); actions
that would move a recorded count or a bit were not applied (`04_ROADMAP.md`). The CPU fingerprint is
exact on 10 scenarios before and after, and exact against the earlier CPU fingerprint.
**Why.** The documents had grown by appending, so one fact could be found in three places with three
ages. The setups table gives the production rule (D-42/D-53/D-64) a name that can be typed and keeps
every other resolution visibly non-comparable.
**Status.** ACCEPTED.

### D-74 — Order of work: CPU-bound items before CUDA-bound items

**Decision.** The CPU-bound close-out runs first — the G1.13 registrations, O-22 convergence, G2.3,
C8, the Li probe, the oracle-silent diagnostic surface and the gate review — then the training
baseline (I10 closed-form `v_s` for one-orbital systems plus the core of I9, the corpus generator for
the E(N) and R scans, with a documented training-target schema). CUDA-bound work — the float32 hot
path with G2.10 and the card-side numerical standard — is grouped for the Windows/CUDA host.
**Why.** The CUDA host has 6 GiB; the memory-bound items need it, while everything else can be
finished on any CPU. The training baseline needs VALID molecular records — H₂ and H₂⁺ records were
INVALID on known-open G1.13 rows — and a label schema before it needs more functionals.
**Status.** ACCEPTED; amends the order of D-63.

### D-75 — G1.13 on molecules: the plain sum aliases the taper shell; lumped mass weights and the lune

**Decision.** Where spheres overlap (every Phase-1 molecule), the far part of the **mass weights** is
the lumped weight `Q_far[f² ℓ_i]` of a temporary lattice refined 2× per axis over the whole box
(`far` and `f²` analytic at the fine points, the cardinal function as three one-dimensional
transposed Lagrange interpolations, blended back to the plain weight within 6 h of a box face); the
far sums of grid fields — `integrate`, `lumped_weights`, the GGA gradient term — keep the plain
`h³ far_i`, and the record carries `far_aliasing`, the relative difference of the two rules on `∫f²`.
The lune — inside sphere `b` but outside sphere `a`, where `β_a Q` had no rule — goes to the sphere
whose nodes cover it (`P_a = Q (β_a + Σ_b β_b [s_b ≥ R_b])`). Node counts, sphere radii, tapers, erf
cells and the angular rule are unchanged; a single centre keeps the plain rule bit for bit.
**Why.** A-11 located the molecular residue in the *sphere* rules; measured with the rules separated,
it was not there. Refining only the far plain sum to `h/2` removes **+2.66e-8** (H₂⁺) and
**+1.81e-8** (H₂): the erf taper `w = 1.3 h` aliases at `exp(−π²w²/h²) = 5.7e-8` of the shell's
content (the module docstring had `exp(−2π²w²/h²) = 4e-15`, the exponent counted twice). A single
shell on a lattice nucleus cancels its own aliasing by symmetry — H at `Z h = 1/4` measures 4e-11
with the same taper — and the second centre's `e^{−2 s_b}` modulation breaks it. The alternatives
fail on memory: densifying the angular nodes (`node_spacing_overlapping` 0.7 → 0.3, 5× the nodes)
still leaves H₂⁺ at 2.5e-8 and 0.5 exhausts the Linux CPU host in the stencil build, and widening
the taper needs `w = 1.5–1.6 h`, `R_s = 13.3 h` (`L`/`Lᵀ` 2.8 → 4.3 GB), out of memory on both
hosts; the lumped weights cost 1.0 s at geometry time (57³) and no solve time. **Mass weights
only:** applied to every far sum they moved the H₂ LDA total by +1.1e-6 Ha (the eigenvalue by
3.4e-6), because `Σ_i ω_i g_i = Q_far[f² I[g]]` interpolates `g`, and a potential with its own cusp
is interpolated badly where the taper's inner tail still carries weight. The lune is 9e-10 on both
molecules. A blend from lumped to plain weights around the spheres was rejected: the degree-7
stencil interpolates an erf of grid-scale width to 4e-3, costing 7e-9 where `f²` is not negligible.
**Measured (CPU, 2026-09-18).** G1.13 at `standard`: `h2plus_R2` 2.290662e-08 → **2.74e-9**,
`h2_R1.4` 2.032007e-08 → **2.93e-9**, the interacting molecules the same to two digits; H, He⁺, He,
He LDA and He⁺ LDA bit-identical to the earlier CPU fingerprint. What remains is the angular Gauss
rule at 0.7 h (±3e-9; 0.6 h gives ≤ 1.7e-9 at 1.35× the nodes) — the threshold stays 1e-8 and the
three rows leave the known-open registry. Energy moves, costs and the A-2 rescan: `02_STATUS.md`
Part A §1–§3, Part B A-2, and the re-bless record below.
**Status.** ACCEPTED, CPU-measured; C8 and O-19 closed; `far_refine` and the edge blend constants are
in the geometry cache key (`CACHE_VERSION` 2).

### D-76 — Oracles off every diagnostic surface unless asked for (O-24)

**Decision.** `CDFT_ORACLES` (through `env_switch`: default off, a typo raises) and `--oracles` on
`run.py`/`test_suite.py` decide whether PySCF/libxc is called at all. Off, the `oracle`-marked tests
(the 30 libxc comparisons of `tests/test_xc.py`) are **deselected**, not skipped, and gates with
`GateSpec.requires_oracle` (G0.6) leave the gate plan and the record rather than printing SKIPPED;
every record says which in `measurements["oracles"]`. G4.7's stored PySCF references
(`cdft.reference.computed`) are data, not an oracle, and stay on; a scenario that names a dropped
gate still gets one SKIPPED row saying why (G5.3).
**Why.** PySCF is absent on the Windows/CUDA host and on the Linux CPU host, so 30 skips and a
SKIPPED G0.6 sat on every summary and every record as noise that hid real skips.
**Status.** ACCEPTED; `tests/test_oracle_switch.py` (22 fast tests) pins the switch, the hook and the
plan.

### D-77 — LOBPCG helpers judge in the operator's measure (O-27)

**Decision.** `LOBPCG.orthonormality_error(hamiltonian, vectors)` and `LOBPCG.residuals(hamiltonian,
evals, vectors)` take the operator and evaluate in `hamiltonian.measure` through one `_measure_of`
used by `solve` as well; the uniform grid measure is the fallback only for an operator without one.
**Why.** On the cusp-factorised path the eigenvectors are φ, orthonormal in `∫f²`; judged in `h³` the
same block reads `max|S − I| ≈ 4e3`, while the solve (D-56) and CheFSI use the weighted measure.
Verified on `h_atom` (21³): helper residuals bitwise equal to CheFSI's, orthonormality 7e-16.
**Status.** ACCEPTED, CPU-verified; the helpers had no callers, so no solver number moves.

### D-78 — The kind is part of a gate's identity: G5.2 EXACT, G1.11 EMPIRICAL, tested in three places (B3)

**Decision.** G5.2 Determinism is **EXACT** in the catalogue, the `GateSpec` and the `03_METHOD.md`
row: the reference is the identity (one configuration solved twice, expected 0) and the 1e-9 Ha
tolerance absorbs float64 reduction-order noise only (2.7e-13 measured, D-61). G1.11 is EMPIRICAL.
`tests/test_gate_catalogue.py` asserts wired kind = catalogue kind for every evaluator and doc kind =
catalogue kind for every row that states one; G5.8 stays the runtime cross-check of lifecycle.
**Why.** Two of 56 gates carried a kind stated differently in code and in the specification with
nothing comparing them; a DERIVED label promises a threshold that tightens with `h`, which a
determinism gate cannot honour.
**Status.** ACCEPTED; G5.8 PASS at 0 disagreements.

### D-79 — `--numerics all-electron` fails G0.2 by construction; documented, not changed (B3)

**Decision.** The component gates run at the spacing of whatever `--numerics` fixes; at the
all-electron placeholder `h = 1` G0.2's fixture (σ = 4h = 4 bohr in its fixed 16-bohr box) truncates
its own test charge at 2σ and reports 5.768e-3 Ha. Recorded in `config.ALL_ELECTRON_NUMERICS`'s note
and the `--numerics` help, pinned by a fast test; `auto` (the default) keeps the component gates at
the model preset.
**Why.** The same solver at `h = 1` in a 64-bohr box passes at 1e-16, so the red row is the fixture's
box, not the Poisson solver; the component gates are scenario-independent, so "resolving the grid
first" has nothing to resolve; and G5.6 is right to exit 1 at the numerics that were asked for.
**Status.** ACCEPTED; no fixture, threshold or grid derivation changes.

### D-80 — Contract 1.6.1: `InversionProtocol` without the copied gate pair (O-29)

**Decision.** `InversionProtocol` drops the `gate_id` / `evaluate(numerics)` pair copied from
`ComponentGateProtocol`; `CONTRACT_VERSION` and `CONTRACT_VERSION_EXPECTED` move to **1.6.1**; no
record field changes, older records stay readable. `runner.UNEVALUABLE_GATE_NAME` is the one string
`known_open.partition_failures` matches, so the `unimplemented` class is reachable (O-26); `_env_flag`
is gone and `CDFT_DETERMINISTIC_WARN_ONLY` / `CDFT_CSR_INT64` read through `env_switch`, a typo
raising (O-28).
**Why.** The protocol is `runtime_checkable`, so a real inverter would have had to expose a gate
interface it cannot honour; the other two are the G5.5 no-hidden-fallback rule applied to the build's
own switches.
**Status.** ACCEPTED; fast tests for each.

### D-81 — Small hosts: the cache budget sees the cgroup, an explicit budget, and visible exclusion

**Decision.** `device_policy._host_ram_bytes` is `psutil`'s total capped by the process's cgroup
memory limit (`cgroup_memory_limit_bytes`, v1 and v2), so the derived CPU geometry-cache budget
(25 %) is 1.46 GiB on a 5.8 GiB host instead of 2.1 GiB; `CDFT_GEOMETRY_CACHE_BYTES` sets an explicit
budget, recorded in `applied_record` and in every record's
`precision_policy["geometry_cache_applied"]`, a bad value raising; `test_suite.py` skips a scenario
whose measured CPU peak exceeds the host (`HOST_MEMORY_NEEDED_BYTES`, today `h2plus_R8_lda` at 8 GiB)
and says so in the console and the JSON (`host_memory_excluded`).
**Why.** On the Linux CPU host (5.8 GiB cgroup limit) the interacting molecular solves peak at
4.4 GB RSS, a full cache beside them is an out-of-memory kill, and `CDFT_GEOMETRY_CACHE=0` is not an
answer for a profile (the cache tests and G5.9's launch budget assume a working cache: 12 failures
and a red G5.9 the first time). `h2plus_R8_lda` on its 116³ box reached 6.1 GB RSS alone and died
before its first SCF iteration; a suite that dies is worse than one that says what it did not solve.
**Status.** ACCEPTED; the `full` profile on that host ran with `CDFT_GEOMETRY_CACHE_BYTES=805306368`
(0.75 GiB), which the JSON records.

### D-82 — O-31: the census reads a memoised CPU-side build as a loop site; host-side implicit syncs

**Decision.** `scripts/sync_census.py` applies its host-side list (`HOST_SIDE_READS`) to
implicit-sync operators as well as to Python reads, and the list gains `gauss_legendre._rule`
(`torch.linalg.eigvalsh` on a CPU tensor plus its `.numpy()`): a site there is reported under
`host_side_reads_total` / `host_side_implicit_total` and never enters the per-iteration counts or the
loop/setup split. The G5.9 row leaves the known-open registry, the budgets of 2026-09-16 stand, and
`test_suite.py` records a non-passing gate's `detail` in the suite JSON.
**Why.** The two exceeded `h_atom` budgets on the Linux CPU host were exactly the 27 Gauss–Legendre
orders the cusp quadrature builds once each at geometry time (`lru_cache`): `loop_syncs` calls any
site with at least `iterations` firings a loop site, and 27 ≥ 7. The rule is built on a CPU tensor
whatever the solve's device — numpy's own LAPACK aborts after `import torch` on the Windows/MKL host
(`03_METHOD.md` Part B) — so it is not a device sync on any host; with it classed host-side the gate
measures **0**, `h_atom` 9/7 reads and 16/7 implicit, `he_scf_probe` 80/12 and 118/12, launches
32 301 / 124 712 — the recorded census to the count. The CUDA host's `full` run of the same night
measured 4 and the JSON kept only that number, which is why a failing gate's detail is now recorded.
**Status.** ACCEPTED, CPU-measured (2026-09-19); `tests/test_sync_census.py` (2 fast tests) pins the
classification; confirmation on the Windows/CUDA host pending.

### D-83 — Census sites are `/`-separated on every platform; four test-side judgements

**Decision.** (1) `SyncCensus._site` returns repository-relative paths with forward slashes on every
platform, so `HOST_SIDE_READS` and the recorded budgets (`src/cdft/...`) match on Windows; before, no
host-side entry ever matched there. (2) `tests/test_device_agnostic.py`'s chunked triangular solve is
judged at `1e-13` of the largest entry (absolute), not `rtol=1e-14, atol=0`. (3)
`tests/test_golden_regression.py` is skipped under `CDFT_DEVICE=cuda`: the pins are the CPU audit
path's, the card is compared by `scripts/fingerprint.py --diff` and G5.2 (D-60). (4) The pure-step
statelessness test compares two steps on a card with non-deterministic reductions allowed at G5.2's
noise (density 1e-11, eigenvalues 1e-10, energies 1e-9 Ha) and bitwise everywhere else. (5)
`test_suite.py` skips a scenario whose measured CUDA peak exceeds the card
(`DEVICE_MEMORY_NEEDED_BYTES`, today `h2plus_R8_lda` at 8 GiB), printed and recorded like the host
exclusion of D-81. No gate threshold, budget or pin changes.
**Why.** Six check runs of 2026-09-19 on the Windows/CUDA host. `pytest -m fast` there: 471 passed,
3 failed — the two `test_sync_census` tests (site strings `tests\test_sync_census.py`) and the
triangular solve (MKL blocks forward substitution by column count; a relative test with `atol=0`
cannot be met by entries near zero — T-4). The same backslashes are why G5.9 measured **4** there
against 2 on the Linux CPU host: on Windows the 27 memoised Gauss–Legendre builds *and* the two mixer
scalars of the host-side list were all counted, on both probes. `test_suite.py --profile full --device
cuda`: unit tests 514 passed, 9 failed — four golden pins at 4e-7 relative on the card (inside D-60's
device agreement, outside a pin), the pure-step bitwise pair, `run.main(["h_atom"])` exiting 1 on the
G5.9 census, plus the three above; `h2plus_R8_lda` ran 631 s to a CUDA OOM at 4.58 GiB allocated on
the 6 GiB card. The golden re-bless there was clean: exactly the ten pins listed for re-blessing
moved (O-30 settled: that host's `p8` is the pin).
**Status.** ACCEPTED; measured on the Linux CPU host (fast tier green); confirmation on the
Windows/CUDA host is the next `pytest -m fast` (expect 0 failed) and `full --device cuda` (expect
unit tests green, G5.9 = 0, `h2plus_R8_lda` skipped visibly, only the registered G2.3 row red).

## Golden re-bless record

Every re-bless of `tests/golden/values.json` must be explained here (`tests/conftest.py`).

| date | what moved (pins) | from → to, or magnitude | why | D-id |
|---|---|---|---|---|
| 2026-09-13 | G3.1 molecular pins | H₂⁺ 7.085e-5 → **1.369e-4**; H₂ 2.700e-4 → **2.111e-4** | the ladder refines instead of extrapolating | D-48 |
| 2026-09-14 | G1.13 pins, new | 1.54e-1 (H, He⁺, He), 1.0e-3 (H₂⁺), 6.1e-4 (H₂) | gate added; fails on all five all-electron rows | D-50 |
| 2026-09-14 | atomic kinetic pins; five G1.13 rows retired | `kinetic.h_atom` 0.49999999980937515 → **0.500000000021** | cusp-aware quadrature | D-54 |
| 2026-09-14 | molecular G1.13/G2.7/G3.1 pins | G1.13 8.1e-8 (H₂⁺), 2.2e-5 (H₂); G2.7 1.6e-5, 4.5e-5; drift → 1.5 | lumped `W` for the D-39 cell average | D-54 |
| 2026-09-15 | molecular eigenvalues; G1.13, G2.7 | eigenvalues by 1.7e-4 and 3.2e-5; G1.13 8.4e-8 → **2.3e-8** and 2.32e-5 → **3.3e-8**; G2.7 H₂ 4.45e-5 → **2.90e-5** | erf-cell spheres | D-58 |
| 2026-09-15/16 | nothing | `pytest --runslow` 264 passed, `tests/golden` untouched | the GPU port moved no CPU number | D-60 |
| 2026-09-16 | `eigenvalue.ground.h2_R1.4`, `kinetic.h2_R1.4` | −1.2842396988406153 → **−1.2842627208563557**; 1.5476477529645032 → **1.5478541811685174** | the lattice rule puts H₂ at `h = 7/30` | D-64 |
| 2026-09-16 | `h2_R1.4` G1.13/G2.7/G3.1 pins | re-blessed on the Windows/CUDA host (the ladder was killed for memory on the Linux CPU host) | the same grid change | D-64 |
| 2026-09-17 | nothing | re-bless on the Windows/CUDA host at commit `e4c905e9a9` (20:16): file identical number for number | confirms the D-64 pins on the card-side tree; D-73 moved no CPU number | D-73 |
| 2026-09-18 | the D-75 pins, on the Linux CPU host (`values.json` **not committed**; the 2026-09-19 re-bless on the Windows/CUDA host is the record of the repository) | `eigenvalue.ground.h2plus_R2` −1.1026280409332716 → **−1.1026281539509817**, `eigenvalue.ground.h2_R1.4` −1.2842627208563557 → **−1.284262876976476**; `kinetic.h2plus_R2` 0.6020922726246347 → **0.602092017961287**, `kinetic.h2_R1.4` 1.5478541811685174 → **1.5478540918101291**; `closed.h2plus_R2.G1.13` **2.7405002320243113e-09** and `closed.h2_R1.4.G1.13` **2.9333252080451904e-09** (new; the `known_open.*.G1.13` pins 2.2906623120568926e-08 / 2.0320073006987452e-08 retired); `closed.*.G2.7` 1.5848245774385816e-05 → 1.5848683090124993e-05 and 1.1890822472659934e-05 → 1.1872479208085984e-05; `closed.*.G3.1` 4.460954466933842e-06 → 4.500057330880658e-06 and 5.7781487777797125e-06 → 5.778426212854626e-06. Everything else moved only in host-dependent last digits (atoms ≤ 2e-15 relative, the radial oracle 4e-12, the fitted stencil orders ≤ 5.3e-6, the self-adjointness pins at round-off) | the lumped far mass weights correct the molecular density normalisation by 1–4e-7 (D-75); the Linux CPU host is a 2-thread torch 2.14.0+cu130 CPU, and its `p8` order is 7.876962576235828 — the Increment-3 value, which settles **O-30**: the pinned 7.876920598620036 is the Windows/CUDA host's | D-75 |
| 2026-09-19 | **the re-bless of record**: the D-75 pins, on the Windows/CUDA host (`pytest --rebless --runslow tests/test_golden_regression.py`, 810 s) | the ten pins listed for re-blessing, at their listed values to 1e-15 (BLAS noise); `known_open.*.G1.13` removed, `closed.*.G1.13` added; `p8` unchanged (O-30 closed: the pin is the Windows/CUDA host's) | the re-bless of record for D-75; the Linux CPU host's copy adopted it number for number | D-75, D-83 |

## Open items

Open items have one home: `02_STATUS.md` Part A §7. This log names the O-ids a decision opens or
closes and keeps no second table; the catalogue, ledger and defect register are likewise in
`02_STATUS.md` (Parts F, B, C).
