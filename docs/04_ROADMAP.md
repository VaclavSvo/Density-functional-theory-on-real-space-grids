# 04 — Roadmap

What comes next, by theme. Sizes are rough ordering estimates, not a schedule. Everything that is
true today is in `02_STATUS.md`; the reasons behind the ordering are D-25, D-63 and D-74 in
`05_DECISION_LOG.md`. Identifiers: `I4`–`I11` are the remaining increments of the technical plan,
`O-nn`/`B-n`/`C-n` are rows of the open-items table in `02_STATUS.md` Part A §7, and `G-x.y` are
gates of `03_METHOD.md` Part C.

---

## 1. Where the project stands

Phase 1 is complete on the CPU: the all-electron, cusp-factorised path for H, He⁺, He, H₂⁺ and H₂
at the non-interacting, LDA and PBE levels, with 46 of 56 gates implemented, a single registered
known-open failure (G2.3 on `h2_R1.4_lda`) and a CPU float64 reference fingerprint over 15
scenarios. The CUDA path reproduces the CPU in float64 to 5e-15 Ha. The one open physics defect is
stretched H₂⁺ at R = 8 (O-22), which does not converge on its derived 116³ box and does not fit a
6 GiB GPU.

The purpose of the instrument is unchanged: gate-certified densities, potentials and energies for
systems where the exact answer is free (one electron) or known to high accuracy, as the reference
a learned exchange–correlation functional or a physics-constrained network is later trained and
judged against. No machine learning enters `src/cdft/` at any point on this roadmap (D-24, G5.7).

## 2. The training corpus and Kohn–Sham inversion (I9 core, I10 one-orbital half)

The first deliverable of the roadmap, because the training baseline needs valid molecular records
and a label schema before it needs a third functional class.

- **Closed-form exact Kohn–Sham potential for one-orbital systems.** For one occupied spatial
  orbital, `v_s = ε + ∇²√n / (2√n)` needs no inversion machinery, and for one electron
  `v_xc = −v_H` exactly. Evaluate on the cusp-factorised grid (the Laplacian taken through the
  analytic factor, never differenced across the nucleus), with a stated density floor and mask;
  compare against `v_ext + v_H + v_xc` of the converged run, against `−Z/r` on the bare hydrogenic
  systems and against the two-centre oracle for H₂⁺. Store under `potentials/v_ks_inverted` and
  `potentials/v_xc_inverted` with `InversionKind.EXACT_ONE_ORBITAL` (D-17, D-31). One new gate
  row, added to the catalogue and to `03_METHOD.md` Part C together (G5.8).
- **Scan records** (D-21, D-22): E(N) for H and He at N = 0.25 … 2.0 in steps of 0.25 (the exact
  reference is the straight line between the integer endpoints; the deviation is diagnostic D1.2,
  recorded and never thresholded), and bond-length scans of H₂⁺ and H₂ on the lattice rule, each R
  at its own `h = R/m` (D-64).
- **A resumable corpus generator** over `CorpusWriter`: outputs named by scenario id and
  `NumericsConfig.fingerprint()`, one HDF5 file per scan, only VALID or MARGINAL records unless
  asked otherwise (G5.4), non-zero exit on any failed point. The full catalogue is 1 188 runs
  (`01_PROJECT.md` Part C §6, about 6.6 GPU-hours); a version-0 cut fits one night on two CPU
  cores.
- **The training-target schema**, as a new section of `03_METHOD.md` Part A §7: for every stored
  field its units, grid, mask, dtype (float64 throughout) and what it labels; the residuals a
  physics-constrained network would minimise (the weak-form Kohn–Sham eigen-residual, the Poisson
  residual, normalisation on the recorded weights, the nuclear cusp condition); and the exact
  constraints that serve as free labels (`E_H + E_xc = 0` and `v_xc = −v_H` for one electron,
  piecewise-linear E(N), the virial). A loader smoke test in pure `h5py`/numpy, never imported by
  the solver.

## 3. Meta-GGA: the gradient rungs, then r2SCAN (O-21)

- The kinetic-energy density `τ` on the sphere nodes (the gradient of `fφ`) and the
  orbital-dependent `v_τ` operator in the weak form; the term `∫ v_τ τ` in the Kohn–Sham energy;
  G0.4/G0.12 evaluated with a non-zero `v_τ`; the fused kernels excluded at this rung (eager,
  recorded).
- The GGA-rung gates first: G1.7 (virial) at GGA by the closed Levy–Perdew form, G1.11 and the
  effective cusp exponent `Z_eff` (D-52 item 3), the G1.8 Lieb–Oxford evaluator fix and the gate
  sub-clauses listed under "gate scope" in `02_STATUS.md` Part A §7.
- r2SCAN [A11]/[A12] transcribed from the primary sources under D-29 (libxc through PySCF verifies a
  transcription, never supplies it), pointwise gates against libxc (G0.6) before any SCF. Every
  registered scenario has `τ = τ_W`, so the self-consistent path exercises only the iso-orbital
  branch; a new EXACT gate on `|τ − τ_W|/τ` states this, and the general-α code is covered by the
  pointwise gates. A record never reports r2SCAN over a PBE density (G5.5).
- Convergence ladder at h = 0.25/0.20/0.16 against PBE's (risk R4); the h = 0.16 rung of H₂ needs
  about 13 GB of host memory.

## 4. GPU: the float32 hot path and the memory work (second half of I6)

- **`hot_dtype` honours `PrecisionConfig.hot_path` on CUDA** (`03_METHOD.md` Part A §6). `f`,
  `f²`, `1/f²`, the mass weights and the similarity-frame preconditioner (D-56) stay float64 and
  are cast per use (`f²` spans 1 … 1e-24; float32 underflows below 1e-38); the divergence form's
  face weights likewise; the Coulomb-cutoff kernel checked float32 against float64 to 1e-6
  relative; the Chebyshev round-off floor measured against the 1e-9 Ritz stop of D-44.
- **G2.10, the mixed-precision audit**, as an artifact gate reading the CPU fingerprint (no
  re-solve): green on all 16 scenarios at 1e-6 Ha/atom and 1e-5 e, the largest deviation per
  quantity recorded. Catalogue row DEFERRED → IMPLEMENTED (count 47/3/6).
- **Memory on a 6 GiB device:** the geometry-cache budget for diatomics (B1: `TRANSPOSE_AS_CSC` or
  a larger `GEOMETRY_CACHE_FRACTION`), `CPU_ONLY_SCENARIOS` for the `gpu` and `full` profiles (B2),
  and the peak of `h2plus_R8_lda` with expandable segments.
- **The GPU-side numerical standard:** GPU-vs-CPU float64 diff ≤ 1e-10 on every scenario, the
  mixed-precision diff, the wall-time targets (`full --device cuda` under 30 min, `quick` under
  2 min), the kernel table `eager` vs `auto` at 57³/81³/96³, and the launch/host-read census that
  backs G5.9. The speed proposals of the 2026-09-17 audit that move float64 bits (reusing the
  subspace apply for the residual, fusing the Chebyshev recurrence on the CPU, one functional
  evaluation per step, the weighted Gram without a temporary) each need a fingerprint re-bless and
  a decision-log entry; the bit-identical ones (dropping discarded residual norms, caching the
  padded Poisson buffer) need a census re-run.

## 5. Stretched H₂⁺ (O-22)

The textbook self-interaction failure — semi-local DFT puts E(H₂⁺, R = 8) 7.5e-2 Ha below E(H)
(G4.5) — is also the one scenario the solver does not converge: a nearly degenerate pair of wells
with one electron. A Kerker trial at the default `q0` did not converge it. Next: `alpha`, `q0` and
level-shift trials with `scripts/probes/stretched_h2plus_mixing_probe.py` on a host with more than
8 GB of memory, after the memory work of §4. Until then `test_suite.py` skips the scenario visibly
on small hosts (D-81, D-83).

## 6. Figures over several records

`figures.py` draws one record. What remains is everything over several: E(N) against the straight
line from the scan records of §2 (figure 1 of the planned paper); the convergence ladders (A-1,
A-10) and the egg-box (A-3); GPU-vs-CPU agreement and the kernel benchmark; a fixture corpus and one
`scripts/make_figures.py` that regenerates the set and exits non-zero on a missing input. The
(R, N)-resolved `v_xc` error field becomes buildable once the closed-form potential of §2 exists.
Nothing on the solve path may import the figure package (G5.7).

## 7. Beyond Phase 1

In order of dependency, not of date:

| id | what | depends on | unblocks |
|---|---|---|---|
| **I4** | Spin polarisation: `n_spin = 2` in the SCF, spin channels in mixing [F23], Fermi and Methfessel–Paxton smearing [F18]–[F20], open shells, atomization energies, H at LSD against NIST; G4.6 | I3 | the flat plane E(N, M); open shells |
| **Li measurement** | Whether the plain cusp factor `exp(−Z r)` survives beyond Z = 2, or the short-ranged factor / pseudopotentials are needed (D-52 item 4) | — | I5 |
| **I5** | ONCV pseudopotentials: UPF parsing, Kleinman–Bylander [E4], double-grid projectors [D16], Fourier filtering [D18]; G0.7, G4.1. Done when He, Be, Ne, Mg valence eigenvalues match SG15 / PseudoDojo to 1e-3 Ha | the Li measurement | chemistry beyond the first row |
| **I7** | Forces: Hellmann–Feynman plus the projector and cusp-factorisation terms; G2.6–G2.8, G3.1, G3.2, G3.5 | I4, I6 | geometry optimisation |
| **I8** | Molecular benchmarks and the cross-code check (Octopus at matched settings, O-2; else GPAW, then PySCF trends): H₂, LiH, N₂, CO, H₂O, CH₄; G4.2–G4.5 | I5, I7 | external credibility |
| **I9** | The corpus at scale: the full 1 188-run catalogue | §2, I4 | training data at scale |
| **I10** | Kohn–Sham inversion for many-electron systems: Wu–Yang [G1], ZMP [G2] with [G4] regularisation | §2 | potential targets beyond one orbital |
| **I11** | Real-time TDDFT over `Hamiltonian.apply` (optional): norm conservation 1e-10, energy conservation under a static Hamiltonian, a known linear-response spectrum | I4 | excited states |

Also open, with no increment of their own: a tighter all-electron reference for interacting H₂ than
the 1.4e-4 Ha basis-set uncertainty of PySCF (O-20; MRChem or a larger basis gives 1e-6), a
cusp-aware interpolant for the interacting density that would reach 1e-6 Ha at production spacing
rather than by refinement (C9, A-10), and the rebuilt pip-only Windows environment that the CUDA
measurements require (B4).

## 8. Out of scope

Hybrid and non-local functionals, periodic boundary conditions, orbital-free DFT, classical fluid
DFT, and any learned component inside the solver (`01_PROJECT.md` Part B §3 and §7). A dependency
that only a learned functional would need is the trigger of risk R10 and needs a decision-log entry
before it enters.
