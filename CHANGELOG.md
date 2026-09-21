# Changelog

All notable changes to `cdft`. The full record of measurements behind each entry is the
improvements catalogue in `docs/02_STATUS.md` (Part F); design decisions are `docs/05_DECISION_LOG.md`.

## 0.5.0 — 2026-09-20 — first public release

Contract 1.6.1. 56 gates catalogued (46 implemented), fast tier 511 tests, one registered
known-open failure (G2.3 on `h2_R1.4_lda`).

Release preparation:

- Repository restructured for publication: documents moved to `docs/`, probe scripts to
  `scripts/probes/`, run artefacts git-ignored except the CPU reference fingerprint
  (`reports/fingerprint_cpu_reference.json`); MIT licence, citation metadata and this changelog
  added; documentation rewritten as project documentation and brought up to date.
- The inner-loop test-suite profile is named `quick` (formerly `agent`); `--gates quick` in
  `figures.py` and `scripts/figures_suite.py` likewise.

Since the last internal milestone (D-75 … D-83, 2026-09-18/19):

- Cusp-weight quadrature exact on molecules: lumped mass weights from a twice-refined lattice
  replace the aliasing plain far sum, taking G1.13 on H₂⁺ and H₂ from 2.3e-8 / 2.0e-8 to
  2.7e-9 / 2.9e-9 against the 1e-8 threshold at unchanged cost (D-75). The molecular records are
  VALID; molecular energies move by 1e-7 to 4e-7 Ha.
- Oracles (PySCF/libxc) off every diagnostic surface unless requested (`CDFT_ORACLES=1`,
  `--oracles`); the setting is recorded in every record (D-76).
- LOBPCG helpers judge in the operator's measure (D-77); gate kinds asserted in three places
  (D-78, D-79); `InversionProtocol` reduced to `kind` and `invert`, contract 1.6.1 (D-80).
- Host memory policy: the CPU cache budget reads the cgroup limit; small hosts run the unit
  tests one file per process and profiles one scenario per process; scenarios that exceed the
  host or the GPU are skipped visibly (D-81, D-83).
- Synchronisation census corrected (memoised Gauss–Legendre rules classed as setup work;
  `/`-separated site names on every platform); G5.9 measures 0 on the CPU (D-82, D-83).
- Golden values re-blessed on the Windows/CUDA host (33 pins, 2026-09-19).

## Before 0.5.0 — internal development, 2026-09-12 … 2026-09-19

The package carried the version 0.4.0 during this period; the milestones below are dated rather
than versioned.

### 2026-09-17 — figures, consolidation, throughput

- Publication figures of one record (`figures.py`, D-72): ten figure types with captions, CSV data
  and a manifest carrying the gate verdict; nothing drawn from an invalid record without a badge.
- Consolidation (D-73): the setups table with `draft`/`standard`/`fine`/`reference` levels,
  `run.py --setups` and `--res`, one-line code comments with decision pointers.
- Throughput on any GPU (D-64 … D-71): the lattice rule (every nucleus on a grid point), fused
  CUDA kernels, the geometry cache, Lanczos bound reuse, CUDA graphs for the Chebyshev filter, a
  card-agnostic device policy and the sync/launch budget gate G5.9.

### 2026-09-16 — GPU port

- GPU port, first half (D-60 … D-62): float64 residency below the SCF, deterministic strict mode
  for the G5.2 pair, the chunked triangular solve around the cuBLAS `trsm` cliff. CUDA float64
  reproduces CPU float64 to 2e-13 Ha on nine scenarios.
- Synchronisation census: one host read per solver iteration.

### 2026-09-15 — self-consistency

- Self-consistency (Increment 3, D-53 … D-58): cusp-aware quadrature with a partition of unity,
  sphere rules and lumped potentials; the interacting grid rule; overlapping spheres split by erf
  cells; the pure `KohnShamStep`, the SCF loop with Pulay/Kerker mixing, nine self-consistent
  scenarios, eleven gates. He at LDA within 1.7e-5 Ha of NIST at production spacing.
- Two oracles: an independent radial Kohn–Sham solver and the prolate-spheroidal two-centre
  solver for H₂⁺.

### 2026-09-12 … 2026-09-14 — foundations

- Skeleton, contract, provenance, HDF5 records, the gate framework (Increment 0).
- Grid, order-2p finite-difference Laplacian, Coulomb-cutoff Poisson solver, CheFSI and LOBPCG
  eigensolvers, the non-interacting solve (Increment 1).
- The cusp-factorised all-electron path (Increment 1b, D-35 … D-42): the Kato factor, the staggered
  divergence form, derived grids, the gate catalogue, the known-open registry and golden values.
  Hydrogenic energies exact to 1e-13 Ha on a 33³ grid.
- The native autograd exchange–correlation layer (Increment 2): Slater, PW92, VWN5, PZ81, PBE,
  B88, LYP, verified against libxc through PySCF to 1e-11.
