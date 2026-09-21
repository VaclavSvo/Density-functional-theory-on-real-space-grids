# 02 — Status

What is true on 2026-09-20. Part A: what works (§1), what to run and the numerical standard (§2),
gate coverage (§3), the validation ladder (§4), limitations (§5), platforms (§6), the project's
single open-items table (§7) and what comes next (§8). Part B is the chemical-accuracy ledger,
Part C the defect register, Parts D and E the two resolved reviews, Part F the improvements
catalogue. Contract **1.6.1**, package **0.5.0**, 56 gates in 7 tiers (46 implemented). Elsewhere:
`01_PROJECT.md` what and why, `03_METHOD.md` how it works, `04_ROADMAP.md` what comes next,
`05_DECISION_LOG.md` the D-entries.

---

## Part A — Status now

Phase 1 is the all-electron path on a uniform grid in the cusp-factorised representation (D-35,
D-38, D-39, D-42), restricted to `Z ≤ 2` and `N ≤ 2`. Two legs: **non-interacting** (bare `−Z/r`,
seven scenarios) and **self-consistent** (Hartree + native LDA/GGA, cusp-aware quadrature D-54,
grid rule D-53, lattice rule D-64 — nine more), each also on CUDA in float64 (D-60).

### 1. What works

Non-interacting leg. The GPU column is a GTX 1660 Ti in float64, "single solve / whole scenario
under the `gpu` profile"; single solves are strict (the 2026-09-15 fingerprint) unless marked *off*.
That GPU is throttled ~2× (D-62).

| system | computed | reference | error | grid | CPU | GPU |
|---|---|---|---|---|---|---|
| H (Z=1) | −0.500000000000 | −0.5 exact | **1.9e-13 Ha** | 33³, h = 1.0 | 0.7 s | 2.5 / 37 s |
| He⁺ (Z=2) | −1.999999999999 | −2.0 exact | **7.8e-13** | 33³, h = 0.5 | 0.8 s | 2.2 / 39 s |
| He (bare, 2e⁻) | −1.999999999999 | −2.0 exact | **7.8e-13** | 33³, h = 0.5 | 0.8 s | 2.2 / 41 s |
| H₂⁺ at R = 2.0 | −1.102628154 | −1.102634217 (oracle) | **6.1e-6** | 57³, h = 0.25 | 21 s (10 build + 11 solve) | 6.9 / 127 s (pre-D-75) |
| H₂ at R = 1.4 | −1.284262877 | −1.284269245 (oracle) | **6.4e-6** | 59³, h = 0.2333 (D-64) | 21 s | 6.3 / 111 s (pre-D-64) |
| harmonic well, 10 levels | (n+3/2)ω | analytic | < 1e-6 Ha | 81³, h = 0.2 | 126 s | 4.5 off (1 state) / 51 s |
| particle in a box | π²n²/2L² | analytic | < 1e-5 rel. | 49³, h = 0.2 | 21 s | 3.7 / 11 s |

Self-consistent leg (`h = 0.25` unless stated, boxes derived per structure; 2 CPU cores, one solve):

| system | functional | computed | reference | error | grid | CPU | GPU |
|---|---|---|---|---|---|---|---|
| He | LDA (VWN) | −2.8348186 | NIST −2.834836 | **1.7e-5 Ha** (A-10) | 57³, box 14 | ~4 min | 13 off / 127 s (45 s quick) |
| He⁺ | LDA | −1.8612293 | NIST −1.861237 | **7.7e-6** | 57³, box 14 | ~3 min | – / 90 s |
| H | LDA | −0.4456693 | NIST −0.445671 | **1.7e-6** (1.24e-4 on box 14) | 84³, box 20.75 | ~8 min | 43 off / 324 s |
| He | PBE | −2.8928984 | radial KS −2.8929349; PySCF −2.892829 ± 1.8e-4 | **3.6e-5**; 6.9e-5 inside the band | 57³, box 14 | ~4 min | – / 129 s (48 s quick) |
| H₂ at R = 1.4 | LDA | −1.1373963 | PySCF −1.137463 ± 1.4e-4 | **6.7e-5** | 96³, h = 0.2333, box 22.17 | 811 s (25 SCF it., 2026-09-18) | – / 467 s (pre-D-64) |
| H₂ at R = 1.4 | PBE | −1.1666019 | PySCF −1.166673 ± 1.4e-4 | **7.1e-5** | 96³, h = 0.2333, box 22.17 | 762 s (26 it.) | 81 s solve (GTX 1660 Ti, 2026-09-17, pre-D-75) |
| H₂⁺ at R = 2 | LDA | −0.5455243 | diagnostic: G4.5 partner, D1.8 | — | 92³, box 22.75 | 317 s (13 it., 2026-09-18) | – / 309 s (pre-D-64) |
| H₂⁺ at R = 8 | LDA | −0.52046862431 (box 14) | diagnostic D1.8: −7.5e-2 Ha below E(H) | unconverged on its derived box (O-22) | 116³, box 28.75 | — | OOM at 200 it. |
| H, N = 0.5 | LDA | −0.2750795 | exact line −0.25 | −0.0251 (D1.2, recorded) | 84³, box 20.75 | 492 s | – / 216 s |

**The models the instrument has** (rows above; VALIDATED = an independent reference and a passing
gate at production settings, DIAGNOSTIC = measured and recorded, never thresholded, D-20). All seven
non-interacting scenarios are validated against exact or oracle values; at **LDA** H, He⁺ and He
against NIST and H₂ against PySCF, while H₂⁺ at R = 2 and R = 8 and H at N = ½ are DIAGNOSTIC
(G4.5, D1.8, D1.2); at **PBE** He (radial KS, PySCF) and H₂ (PySCF). Native but unregistered: PW92,
PZ81, B88, LYP. Runs but has no reference: HeH⁺ (the oracle and the G1.13 closed form refuse a
heteronuclear pair), H₃⁺ (the lattice rule and the partition fall through to their general cases).
**Not built:** spin SCF (I4), pseudopotentials and `Z ≥ 3` (I5, blocked on the Li measurement),
forces (I7, unblocked since D-58), r2SCAN and every meta-GGA (O-21), the float32 hot path and G2.10
(second half of Increment 6), periodic boundaries, Kohn–Sham inversion, TDDFT, the corpus generator
and scans (I9), figures over several records. No machine learning in `src/cdft/` (D-24, G5.7).

### 2. What to run, and the numerical standard

**(a) Commands.** A profile decides *which* checks run and never the settings they run at (several
gates are DERIVED, and a coarsened grid would make them fail correctly). Every profile exits
non-zero if anything fails (G5.6). Raw logs are under `reports/`, which is git-ignored except for
the reference fingerprint.

| command | what it is |
|---|---|
| `pip install -e ".[dev,oracles]"`; `python test_suite.py [--profile full\|gpu] [--device cuda] [--corpus out.h5] [--json …] [--oracles] [--scenarios id …]` | oracles = PySCF, **off by default** (`--oracles` / `CDFT_ORACLES=1` puts the libxc tests and G0.6 back on the plan; the header, JSON and every record say `oracles: off\|on`, D-76); `[dev,figures]` adds matplotlib, `[dev]` alone on Windows. `quick` (the default) = 7 non-interacting + `he_atom_lda` + `he_atom_pbe`, ~15 min on 2 cores; `full` = all 16 with every gate; `gpu` = all 16 on CUDA, strict pair, exit 2 without CUDA. On a host under 8 GiB the unit tests run one file per process and `--scenarios` runs one scenario per process, both printed and in the JSON (D-81) |
| `pytest -m fast` · `--runslow` · `--rebless` · `CDFT_DEVICE=cuda pytest -m fast` · `CDFT_ORACLES=1 pytest -m fast` | the fast tier (511 passed, 1 skipped, ~35 s on 2 cores) · the golden layer and SCF tests · re-pin the goldens · the same on CUDA · the same with the 30 libxc oracle tests, otherwise **deselected** (D-76) |
| `python run.py --setups`; `run.py he_atom_lda --res fine --spacing 0.2 --device cuda`; `run.py h_atom h2_R1.4_pbe [--oracles]` | the setups table (one `Setup` per scenario, levels `draft/standard/fine/reference`, D-73); one scenario at a level; scenarios with their gate tables. `standard` *is* the production rule (D-42/D-53/D-64) and the only level comparable with golden values and the known-open registry; other levels go through `cdft.grid.grid_rule(...)` into `grid_overrides["grid_rule"]`. A model-system level must divide the box edge exactly or `UniformGrid.from_config` snaps `h` and G5.5 flags it; `--res` together with an explicit `--numerics` preset or YAML exits **2** |
| `python scripts/fingerprint.py reports/fp.json [--device cuda] [--deterministic] [--scenarios …]` · `--diff [--tol 1e-10 \| --exact] a.json b.json` | energies at `repr()`, eigenvalues, occupations, SHA-256 of density and orbitals (output path **before** `--scenarios`) · the audit comparison, `--exact` on the CPU and `--tol 1e-10` across devices |
| `python scripts/benchmark.py … --sizes 57 81 96 --fused auto` · `profile_solve.py … --repeat 2` | the kernel table (ms, GB/s, tier) · one solve: wall, peak, allocator counters, build/solve split, cache hit |
| `scripts/`: `sync_census.py`, `scan_throughput.py`, `interacting_ladder.py`, `pyscf_reference.py`, `dissociation_error_scan.py` | launches per solve (G5.9); runs/hour; the NIST ladder (A-10); regenerate the PySCF references; the bond-length error scan (A-2) |
| `scripts/probes/`: `blas_probe.py`, `gemm_probe.py`, `trsm_probe.py`, `cusp_factorisation_probe.py`, `interacting_grid_probe.py`, `stretched_h2plus_mixing_probe.py` | aborting BLAS calls (§6); the GEMM and TRSM kernel cliffs (D-62); the h-independence of the cusp factorisation (A-5); the quadrature table behind D-53; mixing trials for O-22 |
| `python figures.py he_atom_lda [--bond-length 3] [--xc pbe] [--gates full] [--from-corpus …] [--allow-invalid]` · `scripts/figures_suite.py` | solve, gate, record, ten publication figures → `reports/figures/<id>/<run>/` (D-72); an unregistered geometry is built from the nearest registered relative; `--from-corpus` redraws; `--allow-invalid` draws an untrusted record badged, exit 1 while a failure is new; the suite runs all 15 (not `h2plus_R8_lda`) |
| `CDFT_CUDA_GRAPHS=0 CDFT_LANCZOS_REUSE=0 CDFT_GEOMETRY_CACHE=0 CDFT_FUSED=eager` · `CDFT_GEOMETRY_CACHE_BYTES=<n>` · `CDFT_ORACLES=1` | the four throughput levers; each switches off for one attribution run, and the mode that ran is in the record · an explicit cache budget for a small host (D-81, reported as `geometry_cache_applied`; 0.75 GiB is what a 6 GiB host can afford beside a diatomic solve) · the oracle switch (D-76), default off, recorded as `measurements["oracles"]` |

`CDFT_DEVICE={cpu,cuda}` selects the device for the pytest half, `--device` elsewhere; `cuda`
without CUDA is an error, never a fallback (G5.5). **Levels that do not fit a 6 GB GPU:**
`reference` on `h2plus_R8_lda` (181³ = 5.9 M points) and on `h2_R1.4_lda/pbe` (144³ = 3.0 M), and
`fine` on `h2plus_R8_lda` (3.0 M). **Memory:** the slow tier in one process peaks above 6 GB on the
CPU — the geometry cache holds a bundle per geometry and the G3.1 ladders build several; on a small
host run it per file, or set `CDFT_GEOMETRY_CACHE=0`. Runnable without registry entries: any bond
length of H₂⁺, any nuclear charge on the single-centre path, any translation (G2.7), and any
Phase-1 structure at any native functional through `physics_config.interacting(...)`.

**(b) The numerical standard** — the reference numbers every later change is diffed against; each
row is reproduced by the step of (c) in its last column.

| quantity | value (date, where measured) | open |
|---|---|---|
| what defines the standard | the CPU float64 reference fingerprint `reports/fingerprint_cpu_reference.json` together with the golden pins in `tests/golden/values.json`. A change is accepted when it reproduces both, or when the move is attributed digit by digit and the pins are re-blessed | — |
| CPU audit fingerprint | **`reports/fingerprint_cpu_reference.json`** (2026-09-18, Linux CPU host: x86-64, 2 threads, torch 2.14.0+cu130): 15 scenarios, everything but `h2plus_R8_lda` (O-22), at the D-75 tree. The pre-D-75 tree holds the same 15 and reproduced the 2026-09-17 CPU fingerprint (10 scenarios) **bit for bit** (`--diff --exact`); between the two, D-75 moves only the five molecular scenarios (1e-7 to 4e-7 Ha, the density normalisation's aliasing corrected; iteration counts identical). Later work diffs against the reference with `--exact` | `h2plus_R8_lda` (O-22) |
| golden pins (`tests/golden/values.json`) | **33 values** (the two `known_open.*.G1.13` pins retired, two `closed.*.G1.13` pins added). Last move 2026-09-18 (D-75); re-blessed 2026-09-19 on the Windows/CUDA development host (GTX 1660 Ti, torch 2.11.0+cu128, MKL), and that re-bless is the record | — |
| known-open values | registry (`src/cdft/gates/known_open.py`, drift 1.5): **one row**, G2.3 = 1 on `h2_R1.4_lda`, mechanism in its note. G1.13 at `standard`, CPU, 2026-09-18: `h2plus_R2` **2.74e-9**, `h2_R1.4` **2.93e-9**, `h2plus_R2_lda` 2.7e-9, `h2_R1.4_lda`/`h2_R1.4_pbe` 2.9e-9; atoms 8.1e-11 (H) and 1.2e-10 (He, He⁺) unchanged. The CUDA run of 2026-09-18 21:00 UTC reproduces those digits: 2.7405e-9, 2.9333e-9, 2.7822e-9, 2.9103e-9 | — |
| GPU vs CPU, float64 | **max \|Δ\| 4.9e-15 Ha on energies, 2.4e-15 on eigenvalues**, occupations exact, SCF iterations identical, on the 8 scenarios both fingerprints hold (GTX 1660 Ti, 2026-09-16, against the CPU fingerprint of that tree; acceptance 1e-10; different hosts and torch builds, so density hashes differ by summation order). At the Increment 6 tree: 2.05e-13 / 3.27e-13 on 9 scenarios. The single-process CUDA fingerprint runs out of memory on all four interacting molecules (the geometry cache accumulates across scenarios; one process per scenario peaks at 3.74–3.97 GiB and fits) | a CUDA fingerprint of the four interacting molecules, one process each (B1) |
| accuracy at `standard`, GTX 1660 Ti, 2026-09-17 | H LDA **1.68e-6**, He⁺ LDA **7.71e-6**, He LDA **1.74e-5** Ha vs NIST SRD 141 · He PBE **6.89e-5** vs PySCF, 3.65e-5 vs radial KS · H₂ LDA **6.69e-5**, H₂ PBE **7.18e-5** vs PySCF (±1.4e-4) · H₂⁺ bare **6.18e-6**, H₂ bare 1.30e-5 (two electrons) vs the two-centre oracle · `h_atom_N0.5_lda` 1.28e-7 vs radial KS | – |
| solve walls on the GTX 1660 Ti (one solve, 2026-09-17) | `he_plus_lda` 11.6 s · `he_atom_pbe` 14.9 s · `he_atom_lda` 15.7 s · `h_atom_N0.5_lda` 24.6 s · `h_atom_lda` 32.5 s · `h2plus_R2_lda` 37.5 s · `h2_R1.4_lda` 65.1 s · `h2_R1.4_pbe` 80.8 s · bare molecules 10.8–11.6 s | – |
| suite walls | figure suite on the GTX 1660 Ti, 15 scenarios (no `h2plus_R8_lda`): **1093 s** with the `quick` gate budget, **3120 s = 52 min** with `--gates full` (2026-09-17). `test_suite.py`: `quick` ~15 min CPU (2 cores), 174 s CUDA against a 120 s target; `gpu` 3956 s = 66 min against 30 (2026-09-16, before the throughput levers were measured) | `quick` / `full` / `gpu` on CUDA after the throughput levers |
| kernel table and the G5.9 budget | stencil and Chebyshev steps **61–77 GB/s** float64 at 57³–96³, linear in the grid; fused tiers on the CPU only, gemm 104/192 ms vs eager 468/513 ms at 57³/81³. G5.9 PASS at 0, `he_atom_lda` **149 184 launches, 310 syncs, 90 applies** per warm iteration (was 161 262 / 332 / 102). **CUDA `full` run after D-75 (2026-09-18, 3379 s):** 13 of 16 scenarios valid, `h2_R1.4_lda` invalid by the registered G2.3, `h2_R1.4_pbe` **marginal by G3.3 = 0.958** (the last SCF step landed at 96 % of `density_tol`; identical digits on the CPU, so a property of the converged path, not of the device), `h2plus_R8_lda` OOM (O-22); walls `h2_R1.4_lda` 455 s, `h2_R1.4_pbe` 498 s, `h2plus_R2_lda` 240 s, atoms 29–226 s; peaks 4.07 GiB (H₂ LDA/PBE), 3.84 GiB (H₂⁺ LDA) | kernel table `eager` vs `auto` on CUDA; census with the levers on and off |
| test tiers | Linux CPU host, no PySCF: fast tier **511 passed, 1 skipped (CUDA-only), 30 oracle tests deselected** (D-76), ~35 s on 2 cores; slow tier green file by file (§2(a) memory note). `tests/test_golden_regression.py` exceeds a 6 GB host in one process — run its value tests and its two gate tests (`-k gate`) separately. On the Windows/MKL host, 36 numpy-LAPACK tests of `test_gauss_legendre` skip | – |
| geometry and memory | `nucleus_offgrid_bohr` **0.0 on all 14 scenarios** (≤ 2e-15 on H₂). VRAM peaks 2026-09-17: atoms 0.95–1.83 GiB, bare molecules 2.40–2.44 GiB, interacting molecules **3.74–3.97 GiB**; `h2plus_R8_lda` does not fit (OOM at 3.55 + 1.34 GiB, O-22). CPU: the slow tier in one process peaks above 6 GB (geometry cache) | – |
| figure suite | 15 scenarios, **12 exit 0, 3 exit 1** on 2026-09-17 (the then-unregistered G1.13 rows and G2.3); the G1.13 rows now pass and G2.3 is registered, so the suite is expected at 15 exit 0 — **not re-run** since. The kept 2026-09-17 run was made with `--gates full` and carries pre-D-75 numbers | re-run of the figure suite on CUDA |

**(c) The run sequence that sets the standard**, on a CUDA host with the commands of (a):
**0** install, `scripts/probes/blas_probe.py` all `ok`, the fast tier on CUDA; **1** re-bless;
**2** the CPU fingerprint over 16 scenarios (`h2plus_R8_lda` may not converge, O-22); **3** the same
on CUDA, then `--diff --tol 1e-10`; **4** the three D-64 molecules; **5–7** the `quick`, `full` and
`gpu` profiles on CUDA, the last with `--corpus`; **8** the kernel table, `eager` then `auto`;
**9** the census with and without the four levers; **10** the O-22 peak (if it OOMs, add the id to
`CPU_ONLY_SCENARIOS`); **11** scan throughput. Run `nvidia-smi -q -d PERFORMANCE` around 5, 6, 7
and 11. If a target is missed, do not tune: rerun step 5 with one lever off at a time and record
which lever owns the time.

### 3. Gate coverage

**56 gates catalogued in 7 tiers (0–6): 46 implemented, 4 deferred with a named owning increment,
6 out of Phase 1**
(`src/cdft/gates/catalogue.py`; G5.8 keeps specification, catalogue and evaluators in agreement).
Deferred: G2.10 (the float32 hot path, second half of Increment 6); G2.6, G2.8, G6.1 (forces, I7).
Out of Phase 1: G0.7, G4.1–G4.4, G4.6. G1.10 is retired and G1.13 took its place (D-50); G5.9 was
added (D-70); G5.2 was restated as the strict pair (D-69), G1.11 and G2.7 for the lattice rule
(D-64). r2SCAN is deferred as a *functional* (D-55 item 5), not as a gate.

**Known-open failures** — measured, owned, pinned (D-43): they keep their FAIL verdict, their
records stay INVALID, and they do not set the exit code inside their band (drift 1.5).

| gate | scenario | registered | measured since | threshold | item |
|---|---|---|---|---|---|
| G2.3 | `h2_R1.4_lda` | 1 (2026-09-18) | 1 residual rise (5.2e-6 after 4.4e-6 at iteration 22 of 25) inside the five-iteration window; energies fall to 4e-12 Ha | 0 | mechanism in the registry note: the periodic Pulay extrapolation every third iteration overshoots ~20 % in the residual at the converged tail; `h2_R1.4_pbe` has the same rise one iteration earlier, outside its window |

**The molecular G1.13 rows are retired (D-75):** `h2plus_R2` 2.290662e-08 → **2.74e-9**,
`h2_R1.4` 2.032007e-08 → **2.93e-9**, `h2_R1.4_lda`/`h2_R1.4_pbe` **2.9e-9** and `h2plus_R2_lda`
**2.7e-9**, all at `standard` and against the unchanged 1e-8 threshold; pinned under `closed.*`.
The residue was the plain far sum aliasing the taper shell (2.66e-8 on H₂⁺), not the sphere rules;
what remains (±3e-9) is the angular Gauss rule at 0.7 h. **No failure is outside the registry**
since D-75: on the Linux CPU host the `full` gate set exits 0 on the twelve scenarios that fit one
process each (`--scenarios`, D-81) and on the three interacting molecules at the `quick` gate budget
(their `full` gate set — the LOBPCG cross-check on 92³–96³ — exceeds 5.8 GiB), with
`h2plus_R8_lda` skipped visibly (O-22, §7). G1.11 = 229 on `h2plus_R2_lda` is gone: on the D-64 grid
it passes at **5.76e-3**, as D-64 item 5 predicted. Retired rows, with the value they now pass at
(pinned under `closed.*`): G1.13 on H, He⁺, He 1.541933e-1 → **1.16e-10** (D-54) and on the
molecules above (D-75); G2.7 on H₂⁺ 5.56e-5 → **1.58e-5** and on H₂ 1.87e-4 → **2.90e-5**, then
1.189082e-5 at h = 0.2333; G3.1 on H₂⁺ 1.369e-4 → **4.46e-6** and on H₂ 2.111e-4 → **5.78e-6**;
G2.5 1.76e-8 / 1.09e-8 → **2.8e-9 / 7.9e-10** (`full` and `gpu` only). The 2026-09-18
re-measurements of the G2.7/G3.1 pins under D-75 are in the re-bless record (`05_DECISION_LOG.md`).

### 4. The validation ladder

| rung | against | status |
|---|---|---|
| closed forms and identities | harmonic well, box, hydrogenic `−Z²/2`, `∫f² = π/Z³`, `E_H[1s] = 5Z/16`; self-adjointness, transposes, kinetic positivity, scaling (G1.9), lift = interpolateᵀ | round-off (the integrals since D-54) |
| functional derivative · libxc | autograd `v_xc` vs finite difference (G0.5) · every native functional on 10⁵ points (G0.6) | 5.8e-7 worst (LYP) · 1.4e-11 |
| radial oracles | an independent 1-D solver by direct diagonalisation, and its self-consistent version (itself 4e-7 from NIST at LDA) | 4.5e-12 (Z=1) to 8.4e-10 (Z=8); G4.8 passes, He PBE 3-D vs radial 4.5e-5 |
| two-centre oracle · literature | prolate-spheroidal solver at any bond length · Madsen & Peek 1971 through it | 6.2e-6 (H₂⁺), 6.5e-6 Ha (H₂) · 2.2e-9 Ha |
| cusp-weight quadrature | closed-form `∫f²` (G1.13) | atoms 1.2e-10; molecules **2.74e-9** (H₂⁺ R = 2, h = 0.25) and **2.93e-9** (H₂ R = 1.4, h = 0.2333) since D-75, passing; the interacting molecules the same to two digits |
| literature and cross-code, functional | NIST SRD 141 LDA totals; PySCF cc-pV5Z for He PBE and H₂ LDA/PBE inside the stated uncertainty (G4.7) | He 1.7e-5 Ha at h = 0.25, ~O(h³); the PySCF legs pass |
| self-consistency, and the textbook failure | Harris–Foulkes (G2.1, 5e-11), Janak (G2.2), variational tail (G2.3), guess independence (G3.4), SCF criterion (G3.3); the semi-local self-interaction error on stretched H₂⁺ (G4.5, sign only) | passing except G2.3 = 1 on `h2_R1.4_lda`; the textbook failure appears |
| device | GPU float64 vs CPU float64 on nine scenarios | 2.05e-13 Ha; the mixed-precision rung (G2.10) is not built |
| device, after the throughput levers | the same nine with fused kernels, graphs, bound reuse, cache | **pending a CUDA measurement**; on the CPU bit-identical but `he_atom_lda` at 1.8e-15 |
| cross-code | Octopus at matched settings | **out of Phase 1** (I8, O-2) |

### 5. Limitations

* **Interpolation-limited to ~O(h³) on the self-consistent path** (A-10): He at LDA is 1.7e-5 Ha
  from NIST at production and needs `h ≈ 0.16` for 1e-6; the spacing is a cost choice (D-53).
* **The molecular records are VALID since D-75**: G1.13 is 2.74e-9 (H₂⁺ R = 2) and 2.93e-9 (H₂ at
  R = 1.4) against an EXACT 1e-8, with 3× margin; the remaining term is the angular Gauss rule of
  the sphere rules at 0.7 h, and `h2_R1.4_lda` is INVALID on G2.3 alone (§3, a mixer-path artefact).
* **The plain cusp factor is a `Z ≤ 2` method** (D-52 item 4, F4): beyond He it is the short-ranged
  factor or pseudopotentials, decided by a Li measurement not yet made.
* **Spin-restricted only**: H at LDA is the spin-restricted LDA atom (NIST "LDA" −0.445671). And
  **no all-electron oracle for interacting H₂ tighter than 1.4e-4 Ha** (the PySCF basis-set
  uncertainty): agreement inside that band is all G4.7 can assert (O-20).
* **The GPU leg is float64 only.** Strict mode costs ~2×, the GTX 1660 Ti is throttled ~2×, and
  `h2plus_R8_lda` is CPU-only in `full` until O-22 is fixed. Every CUDA number predates the
  throughput levers (§2(b)).
* **Platform.** The audit reference needs a pip-only Windows environment (B4) and the CUDA numbers
  wait on it (§6).

### 6. Platforms and environment

The standing constraints live in `03_METHOD.md` Part A §9: no PySCF wheels on Windows (oracle legs
under WSL2, the stored references making the production path independent of them), **one BLAS stack
per environment** (a conda MKL numpy beside the pip torch aborts the interpreter on every numpy or
scipy BLAS call after `import torch`, OMP Error #15 — B4, `scripts/probes/blas_probe.py`),
`expandable_segments` unsupported on Windows so D-69's allocator half is a no-op there (it matters
for the O-22 peak), and MKL not bitwise reproducible between calls on differently aligned buffers
(defect T-4, Part C). The reference Linux CPU host is 2 cores and a **5.8 GiB cgroup** limit (8 GiB
physical) with no GPU: an interacting diatomic solve peaks at 4.4 GB RSS and tolerates nothing
beside it — not a fast-tier run, not a second build — `h2plus_R8_lda` does not fit at all (O-22),
and the geometry cache must be told the budget (`CDFT_GEOMETRY_CACHE_BYTES`, D-81). The solver's own
`apply_policy` reads the cgroup limit since D-81.

### 7. Open items

The project's single open-items table; `05_DECISION_LOG.md` points here. Ids: `O-nn` open items,
`C-n` the accuracy backlog, `B1`–`B4` the consolidation backlog of `04_ROADMAP.md`, and named rows
for items that never had an id.

| id | item | state / measured | next step |
|---|---|---|---|
| O-2 | Octopus on the target machine (G4.3, G4.4, I8); else GPAW, then PySCF trends | open | the author |
| O-3 | Corpus size and sampling | open; Phase 1 is the 1 188-run scan catalogue (D-22), Phase 2 fewer systems each with an inverted reference (D-17) | the corpus generator |
| O-18 | No licence | **closed 2026-09-20**: MIT, `LICENSE` at the repository root | — |
| O-19 | Cusp-weight quadrature | **closed** (D-54 for atoms, 1.2e-10; D-75 for molecules, 2.7e-9 / 2.9e-9, 2026-09-18) | — |
| O-20 | No all-electron reference for interacting H₂ tighter than 1.4e-4 Ha | open; MRChem or a larger basis gives 1e-6 | a bounded attempt at a tighter reference |
| O-21 | r2SCAN: needs `τ` on the sphere nodes (gradient of `fφ`) and `v_τ` in the weak form | open (D-55 item 5) | the meta-GGA increment |
| O-22 | `h2plus_R8_lda` on its derived 28.75-bohr box (116³ ≈ 1.56 M points) | 200 SCF iterations without converging, then CUDA OOM (3.55 + 1.34 GiB); on the CPU it needs **> 6 GB** (6.1 GB RSS, OOM-killed on the 5.8 GiB host before its first iteration, 2026-09-18), so `test_suite.py` skips it visibly there (D-81) and, since D-83, on any GPU below 8 GiB (the 6 GiB GTX 1660 Ti ran 631 s to a CUDA OOM at 4.58 GiB allocated, twice). **A Kerker trial at the default `q0` (2026-09-18, CPU, 40 iterations, 2549 s) did not converge it:** the residual never falls below 0.04, the energy swings between −0.5205 and −0.480 Ha, and the ladder steps down to damped Pulay at iteration 9 | `alpha`, `q0` and level-shift trials (`scripts/probes/stretched_h2plus_mixing_probe.py`) on a ≥ 8 GiB host; the CUDA memory work first |
| O-24 | PySCF-dependent tests and gates off every diagnostic surface unless `CDFT_ORACLES=1` / `--oracles`; the 56-gate review | **closed** 2026-09-18 (D-76: 30 libxc tests deselected, G0.6 off the plan and the record, `oracles: off` recorded; the gate review corrected 7 catalogue/Part C rows and produced the G1.8 finding below) | — |
| G1.8 | Lieb–Oxford applies `C_LO = 2.273` to `∫n^{4/3}` where D-55 item 4 and Part C state `E_xc ≥ 2.273 E_x^LDA` (the record key `lieb_oxford_integral` holds `∫n^{4/3}`): the bound as coded is 1.35× looser than decided; no verdict changes (E_xc/E_x^LDA is 1.0–1.3 on every Phase-1 system) | open (gate review, 2026-09-18); fix the evaluator to `C_LO · C_x · ∫n^{4/3}` with `C_x = 0.7385587664` and restate the row — a threshold stays 0 | the meta-GGA increment |
| gate scope | Part C sub-clauses no evaluator enforces: G1.2 degeneracy splitting 1e-7, G1.4 monotone decrease under refinement, G2.7 the force clause and the interacting SKIP, G3.2 "mask radius + 2 Å" vs the boxed "edge + 4 bohr", G1.6 the SCF one-iteration form vs the operator identity implemented | open (gate review, 2026-09-18); restate the rows or add the checks | the meta-GGA increment |
| O-25 | `radial_ks` densities inside ~1e-5 bohr carry the Dirichlet truncation at `r_min = 1e-12` | **closed as a stated limit** 2026-09-18: the irregular-solution admixture `~r_min/r`; trust radius 1e-4 bohr in `03_METHOD.md` Part E and the `density_error` caption; totals unaffected at 4e-7 | — |
| O-26 | `known_open.py` matched a string the runner never wrote | **closed** 2026-09-18 (D-80): `runner.UNEVALUABLE_GATE_NAME` shared; the `unimplemented` class is reachable and tested | — |
| O-27 | LOBPCG helpers judged in the grid measure | **closed** 2026-09-18 (D-77): they take the operator and its measure; residuals bitwise equal to CheFSI's on `h_atom` | — |
| O-28 | `_env_flag` weaker than `env_switch` | **closed** 2026-09-18 (D-80): both switches read through `env_switch`; a typo raises | — |
| O-29 | `InversionProtocol` with the copied gate pair | **closed** 2026-09-18 (D-80): contract **1.6.1** | — |
| O-31 | G5.9 counted setup work as loop work, and on Windows matched no host-side site at all | **closed (D-82, D-83)**: the synchronisation census read the 27 memoised `gauss_legendre._rule` builds as a loop site, and named sites with backslashes on Windows so no host-side site ever matched there. With `_rule` memoised on a CPU tensor on every device and sites `/`-separated, the gate measures **0** and both probes sit at their 2026-09-16 budgets exactly (9/7, 80/12); the registry row is retired | — |
| O-30 | `stencil.laplacian.measured_order.p8` pin 7.876920598620036 against a measured 7.876962576235828 | **closed 2026-09-19**: the two figures are the same fit on different BLAS stacks and thread counts, and both lie inside `_FITTED_ORDER_RTOL = 1e-4`. The re-bless left the pin at 7.876920598620036 | — |
| C8 | The molecular G1.13 residue | **closed** by D-75 (it was the plain far sum's aliasing of the taper shell, not the sphere rules; 2.7e-9 / 2.9e-9 at the same node count) | — |
| C9 | 1e-6 vs NIST at production spacing | open by choice: interpolation-limited (A-10), met at `h ≈ 0.16` | whoever spends the compute |
| D-2 | `SOFT_COULOMB` and the cusp factor | **closed** 2026-09-18: the exclusion is stated (Part C, `03_METHOD.md` Part B §6) | — |
| He/lsd | The NIST `He/lsd` row | **closed** 2026-09-18: six terms from the NIST table (Part C) | — |
| Li | The D-52 item 4 measurement (short-ranged factor vs pseudopotentials) | not made; nothing beyond `Z = 2` before it (about two hours of compute, F4) | a two-hour measurement, unscheduled |
| G1.13 rows | the three molecular registry rows | **retired** by D-75 (§3); nothing to register | — |
| G2.3 | `h2_R1.4_lda`: one residual rise in the five-iteration window | **registered** 2026-09-18 with its mechanism (§3); a mixer-path change would move every SCF fingerprint and was not taken | whoever next changes the mixer |
| G2.10 | Mixed-precision audit, float32 hot path | deferred; the seams `as_hot`, `as_accumulate`, `hot_dtype` are in place | the float32 hot path |
| B1 | Geometry-cache budget on a 6 GiB GPU: diatomic bundles 2.0 GiB (int32) against the 1.5 GiB default (25 % of VRAM) | open: `TRANSPOSE_AS_CSC` or a larger `GEOMETRY_CACHE_FRACTION` — `04_ROADMAP.md` | the CUDA memory work |
| B2 | `CPU_ONLY_SCENARIOS` for the `gpu`/`full` profiles, after O-22's memory work | open — `04_ROADMAP.md` | the CUDA memory work |
| B3 | G5.2's kind; G0.2 red under `--numerics all-electron` | **closed** 2026-09-18 (D-78, D-79): G5.2 EXACT in all three places, G1.11 EMPIRICAL, kinds tested; the G0.2 row is the fixture's 16-bohr box at h = 1 truncating its own Gaussian, documented and pinned, not changed | — |
| B4 | The Windows development environment must be rebuilt pip-only (OMP Error #15, §6) | open; no CUDA measurement before it — `04_ROADMAP.md` | the author |
| GPU standard | The "open" column of §2(b); whether a GPU-side G5.9 budget row is wanted | open | the CUDA memory work |
| small items | No Triton wheels on Windows, so `--fused auto` records the failure and takes `gemm`/`eager` | open, cosmetic | the author |
| R4 | Non-local features must be in physical units and spacing-independent, and gated so | open requirement; no feature layer yet (Part D) | the feature layer |
| doc knock-ons | The [A17] corrections of 2026-09-16 ("rung 6", the withdrawn bond-length contrast) | **closed 2026-09-17**: `01_PROJECT.md` Part B §2, `03_METHOD.md` G4.2 and `05_DECISION_LOG.md` D-01 restated. The remaining knock-ons are whatever quotes the entries `00_LITERATURE_SURVEY.md` "Verification status" still lists as unconfirmed | — |

Closed: O-1, O-4, O-5, O-6, O-7, O-8 → D-17, D-25, D-51; **O-9** → D-60–D-62, 2026-09-16 (CUDA
reproduces the CPU to 2.05e-13 Ha); **O-10** → the libxc route through PySCF; O-11 → D-52;
**O-12** → Increment 3, 2026-09-14 (PySCF cc-pV5Z totals with the basis-set difference as the
uncertainty); **O-13** → D-54/D-58 as an accuracy item; O-14, O-16, O-17 → D-35, D-48, D-50;
**O-15** → D-56, 2026-09-14 (G2.5 ≤ 2.8e-9); **O-23** → D-64, 2026-09-16
(`nucleus_offgrid_bohr = 0` on all 14 scenarios); C1/C2 → D-54, D-58; C5 → D-56; C7 → Increment 3;
H4 → D-62 (the cuBLAS `trsm` cliff).

### 8. What comes next

`04_ROADMAP.md` carries the plan and its ordering. The next physics increment is meta-GGA — `τ` on
the sphere nodes and `v_τ` in the weak form, which closes O-21 and r2SCAN — followed by Kohn–Sham
inversion and the training corpus that the PINN baseline consumes. The performance track is the
float32 hot path and the mixed-precision audit (G2.10), the second half of Increment 6, together
with the geometry-cache and allocator work (B1, B2) that the 6 GiB GPU needs. The one open physics
defect, stretched H₂⁺ at R = 8 (O-22), needs both the memory work and a mixing trial beyond the
default Kerker `q0`.

---

## Part B — Chemical accuracy: the standing ledger

Chemical accuracy is 1 kcal/mol = **1.5936e-3 Ha**. Every entry states a gap, gives the measurement
that establishes it, and names the gate that notices when it changes.

**Where the solver stands against the bar** (totals and references in Part A §1): H and He⁺ at
1.9e-13 and 7.8e-13 Ha are 8e-11 and 5e-10 of it; H₂⁺ at R = 2 and H₂ at R = 1.4 are 6.2e-6 and
6.5e-6, **0.4 %** each, where before D-54 they were 64 % and 2.01e-3 Ha; He at LDA is 1.65e-5 at
h = 0.25 and 5.86e-6 at h = 0.20 on box 16 (1 % and 0.4 %), He at PBE 3.6e-5 (2 %). The references
are exact for one centre, `cdft.reference.two_centre` for two (Madsen and Peek 1971 to 2.2e-9 Ha at
R = 2), NIST SRD 141 and `cdft.reference.radial_ks` for the atoms at a functional. **Nothing
currently reported is outside the bar**; against the 1e-6 target of I3 the interacting atoms stand
at 1.7e-5 (A-10) at production and 5.9e-6 at `h = 0.20`.

### A-1 — The multi-centre error is a resolution problem, and resolution fixes it

H₂⁺ at R = 2 against the two-centre oracle under the D-39 cell-averaged `W` (which D-54 replaced),
in the derived 14-bohr box:

| h | 0.50 | 0.40 | 0.32 | **0.25 (production)** | 0.20 | 0.16 | 0.125 |
|---|---|---|---|---|---|---|---|
| points | 29³ | 36³ | 45³ | 57³ | 71³ | 89³ | 113³ |
| error (Ha) | 1.04e-2 | 3.60e-3 | 1.77e-3 | **1.02e-3** | 4.32e-4 | 1.58e-4 | 8.09e-5 |
| × bar | 6.5 | 2.3 | 1.1 | 0.64 | 0.27 | 0.10 | 0.05 |
| wall, 2 cores | 2.6 s | 1.8 s | 4.6 s | 10.5 s | 36 s | 87 s | 315 s |

Monotone, ~O(h³·⁴), no floor: buyable at roughly `n_points^1.3` to `n_points^1.9` (8× the CPU cost
at h = 0.16, 30× at 0.125). **Ruled out:** the Laplacian (orders 2–12 agree to three figures, G3.5
5.2e-8), the box (flat from 12 to 24 bohr, G3.2 2.1e-7), the eigensolver (G2.4 5.8e-7). D-54, D-58
and D-64 bought the error down instead (accuracy history, Part F): 6.2e-6 and 6.5e-6 Ha at
production, G3.1 passing on both molecules. A-1 is a cost statement now; what remained of the family
was A-11, closed by D-75. Gates G3.1, G3.2, G3.5.

### A-2 — The bond-length dependence of the bare H₂⁺ error is flat at the 1e-5 level

Eleven bond lengths against the two-centre oracle (`scripts/dissociation_error_scan.py`), re-run
under D-54/D-58/D-64/D-75 on 2026-09-18 (CPU, `standard`, one derived grid per R):

| R | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 2.0 | 2.4 | 3.0 | 4.0 | 5.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| error (Ha) | +2.29e-5 | +1.43e-5 | +9.04e-6 | +6.58e-6 | +8.27e-6 | +6.11e-6 | +5.38e-6 | +7.30e-6 | +1.03e-5 | +1.40e-5 | +1.51e-5 |
| × bar | 0.014 | 0.009 | 0.006 | 0.004 | 0.005 | 0.004 | 0.003 | 0.005 | 0.006 | 0.009 | 0.009 |

Every point is variational (above the oracle), inside 1.5 % of chemical accuracy, and the shape is
a shallow bowl (largest at the shortest and the longest bonds) with second differences of
1–4e-6 Ha — grid-scale scatter, since each R has its own lattice (D-64), not structure. Before
D-54 the same scan read −5.17 mHa at R = 0.8 falling to −0.37 mHa at 6.0 (3.25× to 0.23× the bar),
smooth and same-signed but 200–1000× larger. Derived observables now inherit ≤ 2e-5 Ha: `D_e`, where
nothing cancels (H₂⁺ dissociates to H + p and the atom is exact to 1.9e-13), is 1.5e-5 Ha from the
oracle, 1 % of the bar. **What closes it:** nothing pending; re-run the scan after the next change
to the partition or the far rule.

### A-4 — Multi-centre convergence is monotone only below h = 0.5

A trap, not a failure. The H₂⁺ error against the oracle at h = 1.00, 0.80, 0.50, 0.40, 0.32, 0.25
is 4.8e-2, **5.2e-3**, **1.0e-2**, 3.6e-3, 1.9e-3, 1.0e-3 Ha — `h = 0.8` beats `h = 0.5`: above `h
≈ 0.5` the error is dominated by where the nuclei fall relative to the grid, and a scan run there
acquires scatter a reader would take for structure — the project's red flag (Q8.1). D-42 fixes the
molecular spacing at 0.25 for this reason, D-64 puts every nucleus on the lattice, and G3.1 reports
monotonicity beside its error.

### A-7 — The Kato cusp cannot be measured at short bond lengths on the production grid

A measurement limitation. G1.11 fits `d(ln n)/dr` around a nucleus, and the window must be local
compared with the distance to the next nucleus or the fit reads the neighbour: before that was
enforced, H₂ at R = 1.4 measured −1.62 against an exact −2.0, a 19 % "violation" that was artefact.
The window is now a fifth of the nearest internuclear distance: H 0.5–3.0 bohr → 2.1e-10; He⁺
0.25–1.5 → 4.8e-12; H₂⁺ at R = 2 0.125–0.4 → 5.76e-3 on the D-64 grid (1.1e-2 linear fit after
D-54); H₂ at R = 1.4 0.124–0.28 → **SKIPPED**, 5 points (6 on the D-64 grid) where 8 are needed. At
GGA the gate must not assert `−2Z` at all (F3): quadratic fit where the window holds four distinct
lattice radii, linear where it does not, SKIPPED with the measured slope (D-55 item 1). **What
closes it:** a finer spacing at R = 1.4, whose cost is a question for after A-10.

### A-10 — The self-consistent energies are interpolation-limited to ~O(h³)

Helium at LDA (VWN) against NIST SRD 141 (−2.834836), `scripts/interacting_ladder.py`:

| h | box | points | E (Ha) | error | fitted order | SCF it. | wall (1 thread) |
|---|---|---|---|---|---|---|---|
| 0.50 | 12 | 25³ | −2.8309185 | 3.92e-3 | — | 17 | 46 s |
| 0.35 | 12 | 35³ | −2.8346404 | 1.96e-4 | 8.4 | 19 | 180 s |
| 0.25 | 12 | 49³ | −2.8348096 | 2.64e-5 | 5.9 | 23 | 267 s |
| 0.20 | 12 | 61³ | −2.8348191 | 1.69e-5 | 2.0 | 26 | 368 s |
| 0.16 | 12 | 75³ | −2.8348211 | 1.49e-5 | — | 47 | 988 s |
| **0.25** | **16** | 65³ | −2.8348195 | **1.65e-5** | — | 27 | 297 s |
| 0.25 | 20 | 81³ | −2.8348196 | 1.64e-5 | — | 28 | 641 s |
| 0.20 | 16 | 81³ | −2.8348301 | **5.86e-6** | 4.6 | 33 | 765 s |

**Box truncation** at 12 bohr is ~1e-5 Ha for He (16 and 20 agree to 1e-7), which is why D-53's
floor is 7 bohr — for helium. Hydrogen is *box*-limited and worse: 1.24e-4 Ha at 14 bohr, 2.32e-6 at
20, 7.7e-7 at 28, `h = 0.20` changing nothing, because its LDA density decays as `e^{−1.37r}`
(ε = −0.233) against helium's `e^{−2.14r}`; D-53 was amended to derive the margin from a
Slater-screened decay estimate (10.4 bohr for H, H₂, H₂⁺; 7 for He, He⁺). **The helium residual:**
the nuclear cusp is integrated exactly (A-9), but the LDA `v_xc ∝ n^{1/3}` has a cusp of its own,
putting an `r³` term into φ that a polynomial interpolant resolves only to O(h³) — degree 5 vs 7
moves the total by 2.4e-5, so it is the interpolant, not the stencil. **Ruled out:** the mixer and
the path (5e-9 with 0, 2 or 4 filter steps; G3.4), the eigensolver (G2.5 2.8e-9), the Hartree split,
the functional (G0.6 1.4e-11; the radial KS oracle hits NIST to 4e-7). **What closes it:**
refinement (`h ≈ 0.16` on box 16, ~4× the cost per SCF step) or a cusp-aware interpolant carrying
the `r³` term. Gates G4.7 (1e-3), G1.7; golden-pinned.

### Closed entries

* **A-11** — the molecular cusp-weight integral at 2–3× the 1e-8 threshold (`h2plus_R2`
  2.290662e-08, `h2_R1.4` 2.032007e-08) → D-75, 2026-09-18: the residue was the plain far sum
  aliasing the erf taper shell at `exp(−π²w²/h²) = 5.7e-8` (not the 4e-15 the module claimed),
  cancelled by symmetry for one centre and not for two; lumped far weights from a twice-refined
  lattice and the lune assigned to the sphere whose nodes cover it give **2.74e-9** and **2.93e-9**
  at the same node count, atoms bit-identical. What remains is the angular Gauss rule at 0.7 h
  (±3e-9; 0.6 h gives ≤ 1.7e-9 at 1.35× the nodes). C8 and O-19 closed.
* **A-3** — egg-box over the force threshold → D-58: G2.7 1.58e-5 (H₂⁺) and 2.90e-5 Ha/atom (H₂)
  against 3.67e-5, 1.189082e-5 on the D-64 grid; with it O-13's last row and C2, unblocking I7.
* **A-5** — the bare path converges as O(h¹) → about the method D-35 replaced; G1.10 retired (D-50),
  measurement in `scripts/probes/cusp_factorisation_probe.py`.
* **A-6** — `E_ext` is the band-energy complement → a defect closed by D-54 and kept as the
  definition: direct and complement agree to ~1e-15 Ha (`external_consistency`, never thresholded).
* **A-8** — LOBPCG stagnated at 291 iterations (solvers agreeing only to 1.8e-8) → D-56: the
  similarity-frame preconditioner gives 2.8e-9 (H₂⁺) and 7.9e-10 (H₂); O-15 retired.
* **A-9** — the density on the derived atomic grids 13.4 % too small and invisible to G1.5
  (`n(0)/n_exact(0) = 0.866`, `E_H[1s]` +0.19 Ha for H and +0.37 for He, grid charge 1.154) → D-54:
  `∫f²` to 1.2e-10, `E_H[1s]` to 2e-9, `E_xc^LDA` to 2e-10, records VALID. Residue: A-11 (closed by D-75).

### What would close this file

A-1, A-3 and their relatives A-2, A-4, A-7 were one defect — the discretisation of the multi-centre
`W` — bought down ~70× at production and, with D-75, to 1e-5 Ha at every bond length. A-11 turned
out to be the *plain* rule's resolution of the taper shell, closed by D-75. What is left open is
A-10 alone: the interpolant's O(h³) on the self-consistent density, closed by refinement or a
cusp-aware interpolant (C9).

---

## Part C — Defect register

The 2026-09-13 pass over every module outside `src/cdft/gates/`, `src/cdft/diagnostics/`,
`contract.py`, `tests/` and `test_suite.py`, plus the test-file defects found on CUDA (T-1–T-4).
Nothing was fixed unless the fix is provably inert to the physics, verified by a full-precision
fingerprint before and after (every energy at `repr()`, every eigenvalue and occupation, SHA-256 of
the density and orbital arrays; all snapshots byte-identical). Known-open gate failures belong to
Part A §3, not here.

**Open.** None.

**Closed.** **D-2** (2026-09-18) — `SOFT_COULOMB` is never cusp-factorised: the cancellation closes
only because `V = −Σ Z_a/s_a` term by term, and a softened potential would leave the singular
residual `Σ Z_a(1/s_a − 1/√(s_a²+c²))` on the untransformed side; `_use_cusp_factorisation`
selects `NUCLEAR_COULOMB` only and `03_METHOD.md` Part B §6 states it. **D-5** (2026-09-18) — the
NIST `He/lsd` row carries all six terms from the NIST table (a closed shell: identical to `He/lda`
term by term, which the test now asserts instead of the gap); the residual bound of 2.5e-6 is kept,
the residuals `E_tot − (E_kin + E_coul + E_enuc + E_xc)` being H/lda 1.000e-6, H/lsd 5.6e-17,
He/lda, He/lsd and He⁺/lda 0, He⁺/lsd 1.000e-6, five printed values each rounded to 0.5e-6
(`tests/test_literature_values.py`). R-1 the p8 golden pin failed on Windows →
`_FITTED_ORDER_RTOL = 1e-4`, 20× the measured 5.3e-6 cross-platform spread, no re-bless. R-2 G1.10
failed → retired (D-50). D-1 `0 tests` reported → a duplicated `-q` dropped the summary line.
D-3 `precision.as_hot` / `as_accumulate` unreferenced → kept as the float32 casting seam.
D-4 `FILE_CONTRACT_VERSION` dead → removed, the version stamped through provenance. D-6 a cited
test file did not exist → `tests/test_divergence_form.py` exists. D-7 stale duplicates of the
contract and of an output folder → deleted. D-8 cosmetic and structural items → fixed and inert,
the Lanczos factoring keeping `inner`/`norm` parameterised because `grid.inner` and `Measure.inner`
sum in different orders; the contract's `__all__` named an undefined `ComponentGateProtocol` →
fixed by D-49. T-1–T-3 device attributes compared against a bare `cuda` fixture → it returns
`cuda:0`. T-4 chunked CPU BLAS asserted bitwise, which MKL on Windows does not guarantee → the test
now spies one full-width solve and compares to `rtol = 1e-14`; it was the one failure in the first
CUDA run of that tree (322 passed / 1 failed) and is green since.

---

## Part D — Design review of 2026-09-13, as resolved

Six findings, resolved by D-47 (scope: exact KS solutions for molecules; training is a separate
project), D-48 (the multi-centre error is a resolution problem, which this review stated wrongly),
D-49 (contract 1.5.0) and D-52.

| # | finding | resolution | status |
|---|---|---|---|
| R1 | Gradient through the SCF | Not owed here (D-47); autograd *inside* the functional is provided (G0.5). What survives is `SCFStepProtocol`, the mixer-free pure `step`/`residual` pair, because unrolling is memory-infeasible at 50 iterations, degree 20, fp32 (He 0.13 GB, H₂⁺ 0.69 GB, H₂O at h = 0.25 **7.9 GB**, a 100-atom target **559 GB**) | DESCOPED; contract 1.5.0 |
| R2 | — | Withdrawn: a stated scope exclusion read as a missing feature | WITHDRAWN |
| R3 | `E_ext` could be silently mixed in an append-only corpus | Every record carries `external_method` | IMPLEMENTED |
| R4 | Per-scenario grids are correct physics and a trap for grid-based learning | Requirement stated, gate owed | **OPEN** (§7) |
| R5 | Fractional occupations and spin, the spin axis, the `apply` seam, G5.7, corpus contents, contract versioning | Checked and sound: `ScanKind` already carries the fractional-charge, fractional-spin and flat-plane axes, the spin axis has extent 1 from I1, G5.7 does not obstruct training, and every field a learned functional needs is stored per record | SOUND |
| R6 | The singular `∫n·v_ext` | Superseded by A-9: the standard total-energy form never needs it (D-52 item 2, F2) | SUPERSEDED |

**R4, the one still open.** D-42 derives the grid from the structure, worth 250× in speed, so every
scenario has a different grid (`h_atom` 1.00/33³, `he_plus` 0.50/33³, `h2plus_R2` 0.25/57³,
`h2_R1.4` 0.2333/59³). For **semi-local** training that is an advantage: the functional is evaluated
pointwise, every grid point is an independent sample, and spacing diversity guards against a model
that has memorised one discretisation. For **non-local** features it is a trap — a CIDER-style
Gaussian convolution at h = 1.0 and at h = 0.25 does not give the same feature for the same density
unless it is built in physical units and convergence-tested — and a blocker for any architecture
consuming the grid as a tensor. Gate it when the feature layer is built: same feature, two spacings.

---

## Part E — The 2026-09-14 review

**Verdict.** The literature review is sound and its target right — self-interaction and flat-plane
error, where exact labels are free. The instrument was one-third built at the time: Phase 1 at
round-off, with Hartree, XC, self-consistency, spin, pseudopotentials, forces and GPU unbuilt. Both
destinations under discussion (SIE physics on `Z ≤ 2`; D-47's chemical-accuracy molecules) shared
the same next two increments, so there was no ordering decision; I2, I3 and I6 are delivered
(Part F). The checks behind F1, F3 and F4 were made numerically.

| # | finding | consequence | status |
|---|---|---|---|
| F1 | The derived atomic grid (D-42) is unusable for interacting systems: `E_H[1s]` wrong by +0.19 Ha (H) and +0.37 (He) at `h = 1/Z`, grid charge 1.154, ~O(h⁴) in `Zh`; at h = 0.25 He's Hartree energy alone is 14× chemical accuracy | Re-derive the grid rule and integrate every density integral with analytic `f²` and interpolated `φ²`; `∫n = N` to 1e-10 is the sharpest gate | **closed** by D-53 + D-54 (A-9), with the Hartree split `n = Σ c_a e^{−2Z_a s_a} + n_smooth` |
| F2 | R6 is bookkeeping: the standard total energy contains no `∫n·v_ext` | `E_ext` stays the exact complement; what survives of R6 is F1 | **closed** (D-52 item 2, A-6) |
| F3 | GGA potentials diverge at the nucleus. Fitting `v ≈ a/r + b` for r < 0.03 on a hydrogenic density: LDA (PW92) a = 0.000, b = −0.739; PBE exchange −0.044/r; PBE correlation +0.027/r; PBE total **−0.017/r**; B88 −0.046/r, so under PBE the orbital sees `Z_eff = Z + 0.017` | G1.11 must stop asserting `−2Z` at GGA level; expect O(h) back with a prefactor ~50× smaller than A-5 (≈ 5e-5 Ha at h = 0.16) | **acted on** (D-55 item 1); drawn by `figures.py` |
| F4 | The plain factor cannot leave the first row: `φ = ψ/f` grows as `e^{(Z−κ)r}` — hydrogenic 2p at Z = 10 gives 1.5e2 at 1 bohr, 4.4e4 at 2, **1.9e9 at 4**; Li's 2s (κ ≈ 0.45) reaches e^{25} at 10 bohr; He (κ ≈ 1.07) is fine | The real reason I5 matters; a cone-free short-ranged factor (`u = Zs g(s²)` with `g′(0) = 0`, e.g. `g = exp(−s⁴/a⁴)`) would extend the path to the second row | **open** — the Li measurement (§7) |

**Problems with the problem itself** (not fixable by code). *Transferability*: H, He⁺, H₂⁺, H₂ are
where SIE is cleanest, not most representative, so the catalogue must grow to open-shell atoms and a
charge-transfer pair (LiF) once spin exists — F4 decides all-electron or pseudopotential.
*Confirmation vs discovery*: the flat plane on a grid is a confirmation; the result lives in the
spatially resolved `v_xc` error over (R, N) and in the diffuse-density regime, and kill-check 3
(PySCF at large basis vs grid for E(N = 0.9)) decides whether the basis-freedom argument holds,
before the scan machinery. *Endpoints*: E(H⁻) = −0.527751 Ha is transcribed with its citation, and
the H₂ N = 2 endpoint needs an exact-quality density (FCI/cc-pV5Z gives both to ~1e-4).
*Labels for general chemistry* come from outside: the plan names the set it joins.

---

## Part F — Improvements catalogue

One row per milestone, chronological; decisions are in `05_DECISION_LOG.md`.

| milestone | delivered | measured effect | decisions |
|---|---|---|---|
| **2026-09-12 — Increment 0** | Repository, pinned environment, the contract as the API, schema-validated config, provenance, HDF5, the gate framework, CI | G5.1 0 missing fields; G5.3 0 incomplete entries; G5.6 exercised; G5.7 0 violations over 36 modules, tested against a planted violation (`test_contract_and_records.py`) | D-15, D-27–D-31 |
| **2026-09-12 — Increment 1** | Grid, order-2p Laplacian, Coulomb-cutoff Poisson, CheFSI and LOBPCG, the non-interacting solve, gate tiers 0–2 | G0.1 order **7.877** on \|r\| ≤ 3σ (whole-box fit −2.74: the zero-filled halo enters ∇²f/h²); G0.2 4.4e-16 Ha at pad 2.5, where pad 2.0 leaves 1.15e-4 relative in V the energy cannot see; G1.2 7.06e-7 Ha at h = 0.20, order **8.04**; G1.3 exact to 8 decimals after `ODD_REFLECTION`; G1.6 8.9e-14; G2.5 4.2e-9 in 158 LOBPCG iterations; **G1.10 FAIL** 2.77e-3 (H), 1.51e-2 (He⁺), fitted q = 1.05 / 0.62 against a nominal 8. 60 tests; CheFSI 10 states 133 s at 81³ | D-32, D-33; pad factor 2.5; a gate reports the error at its finest spacing, never an extrapolation |
| **2026-09-13 — Increment 1b, cusp factorisation** | The cusp-factorised all-electron path, unplanned: factor, staggered divergence form, cell-averaged `W`, the grid rule, both oracles, the gate catalogue, the known-open registry, goldens | Hydrogen **1.8e-3 → 1.9e-13 Ha in 0.7 s**; the probe returns the exact energy to 1.3e-7 (H) and 1.7e-10 (He⁺) independently of h, a smooth basis converging at order 8.2 / 7.4. 145 fast tests, 34 of 55 gates; G1.13 then failing on all five all-electron scenarios (D-50 → A-9) | D-35–D-43 |
| **2026-09-14 — Increment 2, XC layer** | The native autograd XC layer: Slater, PW92, VWN5, PZ81, PBE, B88, LYP, dispatch, the libxc oracle, five gates | G0.6 on 10⁵ points × {polarised, not} × {s ≤ 3, s ≤ 8} × 15 functionals; G1.1 3.3e-16; G1.8 < 2.273; G1.9 round-off; G0.5 and G0.6 values in §4. 55 tests; PZ81's 3.2e-5 seam at `r_s = 1` reproduced | D-55 items 3–5; r2SCAN → O-21 |
| **2026-09-14 — review fixes (F1–F4)** | The four measured findings of Part E and the documentation cut | F1 set the I3 target (+0.19 / +0.37 Ha, grid charge 1.154); F3 restated G1.11; F4 put the Li measurement before I5 | D-47, D-48, D-49, D-52 |
| **2026-09-14/15 — Increment 3, self-consistency** | The cusp-aware quadrature, the pure `KohnShamStep`, the SCF loop, `solve_scenario`, the interacting grid rule, two oracles, nine scenarios, eleven gates | Atoms `∫f²` **1.541933e-1 → 1.16e-10**, `E_xc^LDA` to 2e-10, amplitude ratio 0.866 → 1.000; molecules G3.1 1.369e-4 → **5.35e-6** and 2.111e-4 → **8.84e-6**; H₂ LDA/PBE 1.74e-4 / 1.65e-4 from PySCF on box 14. Fast tests 175 → **240**. Every digit of the energy moves attributed by a before/after fingerprint | D-53, D-54, D-55 items 1–2, D-56, D-57 |
| **2026-09-15 — Increment 3 addendum** | Backlog C2: overlapping spheres with full tapers, the overlap split by erf cells of width `h`, angular nodes at `0.7h` | Egg-box, `∫f²` and both molecular energies in the accuracy history; atoms unchanged. G3.1 on H₂ *rose* 8.8e-6 → 2.8e-5 because the squeezed-taper error had cancelled between ladder rungs — the gap is now the error. Dead ends measured (Becke cells leak into a lune, 2.4e-6; confining them switches sharply, 2.2e-5) | D-58 |
| **2026-09-15/16 — Increment 6, first half (GPU port)** | The device seams of `precision.py`, residency below the SCF, the contract's device block, the GPU legs of G5.1 and G5.2, `--device`, the diagnostic scripts | **GPU float64 reproduces CPU float64 to 2.05e-13 Ha** on nine scenarios against a 1e-10 acceptance (§2), occupations exact, iterations identical. After the `trsm` fix `harmonic_w1` 593 → **4.5 s**, `h_atom_lda` 1878 → **43 s**, `he_atom_lda` 30.9 → **13 s**. G2.5 ≤ 4.2e-9 on CUDA; G4.7 on H₂ 1.74e-4 → **6.7e-5** (LDA), 1.65e-4 → **7.7e-5** (PBE). No CPU number moved (264 slow tests green) | D-60, D-61, D-62; O-9 closed, O-22 and O-23 opened |
| **2026-09-15/16 — synchronisation census** | The host-synchronisation census and the CPU cleanup it drove: one host read per solver iteration, small tensors on the CPU | Host reads `he_atom_lda` 3521 → **332**, `h_atom_lda` 4165 → **421**, `h2plus_R2` 2699 → 88; launches 247 665 → 161 262 and 314 803 → 208 726. CPU walls ~2× faster with no number moved (`harmonic_w1` 78.9 → 40.7 s, `h_atom_lda` 1062 → 466 s); the CPU fingerprint stayed bit-identical. Host cost = reads × 1.3 ms + launches × 61 µs | D-60 night addendum |
| **2026-09-16 — throughput work (I6b)** | Throughput on any GPU, seven parts, CPU-verified; CUDA numbers pending (§2) | Fast tests 237 → **323**; the CPU fingerprint identical on the five non-interacting scenarios, `he_atom_lda` moving 1.8e-15 Ha on the bound-reuse path (bit-identical with the lever off); G5.9 PASS at 0 | D-64–D-71 |
| ↳ lattice rule | `h = d/m`, the edge rounded up, the lattice on the first nucleus | `nucleus_offgrid_bohr` **0.0 on all 14 scenarios** (was 0.054 / 0.044 / 0.0037 / 0.0012); H₂ → 59³ at `h = 0.2333` and **6.5e-6 Ha**; `h2_R1.4_lda/pbe` 96³, `h2plus_R2_lda` 92³, `h_atom_lda` 84³ at 0.25 | D-64, closes O-23 |
| ↳ fused kernels | Fused device kernels, tiers chosen by a first-use probe (`CDFT_FUSED`) | CPU divergence gemm 104 / 192 ms vs eager 468 / 513 ms at 57³ / 81³ | D-65 |
| ↳ geometry cache | The byte-bounded geometry cache, `bypass()` for G5.2/G5.9 | `h_atom` twice: 2.86 s (build 1.19 + solve 1.68) then **1.25 s** (build 0.00), bit-identical | D-66 |
| ↳ Lanczos bound reuse | Lanczos bounds reused across SCF iterations, widened by the Weyl shift (`CDFT_LANCZOS_REUSE`) | launches 161 262 → **149 184**, syncs 332 → **310**, applies per warm iteration 102 → **90** | D-68 |
| ↳ device policy | `DeviceProfile` → `DerivedSettings`, TF32 asserted off; **G5.9** (`device_policy.py`) | Catalogue 55 → **56 gates, 46 implemented**; G5.9 PASS at 0 | D-70 |
| ↳ graphs, strict pair, throughput scan | `GraphedChebyshevFilter` (one CUDA graph per block shape, validated against eager once); the strict pair with expandable segments, memory released between solves and `CPU_ONLY_SCENARIOS`; `scan_throughput.py` for runs/hour at workers or streams with peak VRAM | all pending a CUDA measurement; the allocator lever is a no-op on Windows (§5) | D-67, D-69, D-71 |
| **2026-09-17 — figures** | `figures.py` at the root: scenario selection or composition, solve and gate, a field evaluator through the cusp, Becke-fused integration grids, ten figures with applicability tests, PDF/PNG/SVG with captions, CSV and a manifest; `scripts/figures_suite.py`; `tests/test_figures.py` | Fast tests 323 → **448**; 95 figure tests, 91 s. Pinned: exact 1s `n`, `∇n`, `∇²n` to 1e-10 relative through the cusp; the GGA `v_xc` against the radial formula to **8e-16**; hydrogen's reduced-gradient median 1.1543946 vs 1.1543907; `E_H` of the 1s density to 2e-6 and of a pair to 1.2e-8. On the D-64 grids `h2plus_R2_lda` G1.11 **5.76e-3**, G1.13 2.290e-8, G2.3 = 0; `h2_R1.4_lda` G1.13 **2.032e-8** and **G2.3 = 1 survives the lattice rule**; D1.1 of H₂⁺ at LDA 0.0555348 Ha on two quadratures (to 7e-10); D1.6 +0.333, +0.324, +0.562, +0.297, +0.047 Ha (He LDA, He PBE, He⁺ PBE, H₂⁺ LDA, H at N = ½) | D-72; O-25 opened, O-24 listed |
| **2026-09-17 — consolidation** | Six documents to one purpose each, open items in §7 only and the standard in §2 only; code comments cut to one-line summaries plus `D-nn` pointers; the setups table, `cdft.grid.grid_rule`, `run.py --setups/--res`, `figures.py --res`; dead code removed and the bit-identical speed actions applied (`UniformGrid.partial_derivative` in the GGA divergence, the wasted clone and spin reduction in `KohnShamStep.potentials`, the memoised golden solves); the base-preset rule in one place (`config.numerics_for`) | Documentation **915 → 390 KB**; code 34 643 → **32 033 lines** (comments and docstrings 8 437 → 5 166 lines, 24 % → 16 %); one gradient component instead of three at 60³: 4.11 → **0.46 ms** per call; golden value tests 62 → **35 s**; fast tests 448 → **503**. **No number moved:** CPU fingerprint exact on 10 scenarios before and after, and exact against the previous CPU fingerprint. O-26–O-30 opened | D-73; D-74 |
| **2026-09-18 — G1.13 closure (D-75)** | **G1.13 fixed on every molecule** — lumped mass weights from a twice-refined lattice with the lune assigned to its sphere replace the aliasing plain far sum (D-75); G2.3 on `h2_R1.4_lda` diagnosed and registered, leaving the registry at that one row; oracles off every surface (D-76); O-26–O-29 and B3 fixed with tests (D-77–D-80, contract **1.6.1**); the 56-gate review (7 rows corrected, G1.8's constant convention found); He/lsd, D-2 and O-25 closed; A-2 re-scanned | G1.13 `h2plus_R2` 2.290662e-08 → **2.74e-9**, `h2_R1.4` 2.032007e-08 → **2.93e-9**, interacting molecules 2.7–2.9e-9, at the same node count and solve cost (+1.0 s build); atoms bit-identical; molecular energies move 1e-7 to 4e-7 Ha (bare H₂⁺ vs the oracle 6.18e-6 → **6.06e-6**, H₂ LDA vs PySCF 6.69e-5 → 6.66e-5); A-2 at every R inside 1.5 % of chemical accuracy (was up to 3.25×); fast tests 503 → **503 + 22 + 9**, 30 oracle tests deselected | D-75–D-82 |
| **2026-09-19 — census and platform close-out** | O-31 and O-30 closed; the CUDA `full` run confirmed the G1.13 digits on the GTX 1660 Ti; the Kerker probe ruled out the default-`q0` fix for O-22; the GPU memory exclusion and four test-side judgements settled | G5.9 measures **0** on the Linux CPU host with both probes at their 2026-09-16 budgets exactly (9/7, 80/12); the re-bless left `p8` at 7.876920598620036, ten pins clean; `h2plus_R8_lda` excluded below 8 GiB of VRAM | D-82, D-83 |

### Performance history

| date | `quick` profile | `full` / `gpu` profile | `he_atom_lda` solve | launches per solve | note |
|---|---|---|---|---|---|
| 2026-09-18 | CPU, 2 cores | `full` on the CPU, one scenario per process | 65 s CPU (Linux CPU host) | — | D-75 adds 1.0 s to a diatomic build; solves unchanged (`h2plus_R2` 10.8 → 10.5 s solve) |
| 2026-09-14 | ~15 min CPU (2 cores) | full CPU ≈ 5 h | 199.6 s (1 thread) | 247 665 | Increment 3 |
| 2026-09-15 | — | — | 104.8 s CPU; 30.9 s CUDA | 247 665 | the synchronisation census: CPU ~2× faster, no number moved |
| 2026-09-16 | **174 s** CUDA (target 120) | `gpu` **3956 s = 66 min** (target 30) | **13 s** CUDA (off) | 161 262, then **149 184** after the graph and bound-reuse work (syncs 310, 90 applies/it) | after D-62 and the CPU census; `full --device cuda` never measured |
| 2026-09-17 | figure suite on the GTX 1660 Ti, 15 scenarios: **1093 s** (`quick` gates) | the same with `--gates full`: **3120 s = 52 min**; `test_suite.py` profiles not yet re-measured after the throughput work | **15.7 s** CUDA | — | consolidation moved no number (fingerprint exact) |

### Accuracy history

| change | quantity | before | after |
|---|---|---|---|
| D-75 (2026-09-18) | G1.13 on H₂⁺ R = 2 / H₂ R = 1.4 / the interacting molecules | 2.290662e-08 / 2.032007e-08 / 2.03–2.29e-8 (known-open) | **2.74e-9 / 2.93e-9 / 2.7–2.9e-9** (pass) |
| D-75 | H₂⁺ R = 2 bare vs oracle; A-2 scan R = 0.8 … 6 | 6.18e-6; −5.17 mHa … −0.37 mHa (pre-D-54) | **6.06e-6**; +2.3e-5 … +5.4e-6 Ha |
| D-54 (2026-09-14) · D-58 (2026-09-15) | H₂⁺ R = 2 vs oracle | 1.02e-3 · 2.6e-5 | 2.6e-5 · **6.2e-6** |
| D-54 · D-58 · D-64 (2026-09-16) | H₂ R = 1.4 vs oracle | 2.01e-3 · 1.45e-4 · 3.0e-5 | 1.45e-4 · 3.0e-5 · **6.5e-6** at `h = 0.2333` |
| D-54 | G1.13 on H, He⁺, He; `E_H[1s]` at `h = 1/Z` | 1.541933e-1; +1.9e-1 Ha (H), +3.7e-1 (He) | **1.16e-10**; 2e-9 |
| D-58 · D-64 | G1.13 on `h2_R1.4`; egg-box G2.7 on H₂ | 2.32e-5 · 3.317796e-08; 4.45e-5 (fail) · 2.90e-5 | 3.32e-8 · **2.032007e-08**; **2.90e-5 pass** (threshold 3.67e-5) · 1.189082e-05 |
| D-53 amendment | H at LDA vs NIST | 1.24e-4 (box 14) | **1.7e-6** (box 20.75) |
| refinement (A-10) | He at LDA vs NIST | 1.65e-5 at `h = 0.25` | **5.86e-6** at `h = 0.20` |
| D-56 | G2.5, CheFSI vs LOBPCG | 1.76e-8 / 1.09e-8 | **2.8e-9 / 7.9e-10** |
| D-64 | G1.11 on `h2plus_R2_lda` | 229 | **5.76e-3** (passes) |
| D-58 + derived box | G4.7 on H₂ LDA / PBE vs PySCF | 1.74e-4 / 1.65e-4 | **6.7e-5 / 7.7e-5** (inside ±1.4e-4) |

### How to add to this file

Update Part A in place (§1, §2, §3, §7) and add or close a Part B entry if the accuracy picture
moved. Add **one row** to the catalogue above — milestone, delivered, measured effect, decisions —
and the decisions to `05_DECISION_LOG.md`. Never append a narrative.
