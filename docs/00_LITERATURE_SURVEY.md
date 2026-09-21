# 00 — Literature Survey

**Classical DFT Solver** — the non-neural Kohn–Sham baseline for PINN training · V. Svoboda, WPI
Vienna · 170 tagged entries · source of record [A17] · last full verification pass 2026-09-16.

Each entry is a bibliographic record plus `**[USE]**`: what this project takes from it — a formula,
a number, a design rule, a gate, a module; no `[USE]` means it was read and rejected, with the
reason. Sections A–K group by subsystem, so each part of `src/cdft/` has an anchor; L reads the
machine-learned-functional literature back onto this project. Tags are an API: code and the other
five documents cite them, so they are never renumbered or removed. To add an entry — the next free
number in its section, the full record with its DOI, a `[USE]` of at most two lines.

---

## A. Foundations and the exchange–correlation functional

**[A1]** P. Hohenberg, W. Kohn. *Inhomogeneous Electron Gas.* Phys. Rev. **136**, B864 (1964).
DOI: 10.1103/PhysRev.136.B864
**[USE]** The ground-state energy is a functional of n(r) alone — the reason the corpus stores
n(r), not wavefunctions, as the primary training artefact.

**[A2]** W. Kohn, L. J. Sham. *Self-Consistent Equations Including Exchange and Correlation
Effects.* Phys. Rev. **140**, A1133 (1965). DOI: 10.1103/PhysRev.140.A1133
**[USE]** The equations the solver implements. Fixes the energy decomposition E = T_s + E_ext +
E_H + E_xc + E_ii, which is reproduced term by term and stored per record.

**[A3]** J. P. Perdew, K. Schmidt. *Jacob's Ladder of Density Functional Approximations for the
Exchange-Correlation Energy.* AIP Conf. Proc. **577**, 1 (2001). DOI: 10.1063/1.1390175
**[USE]** The rung taxonomy `xc/` is organised by — rung 1 needs n, rung 2 adds ∇n, rung 3 adds τ
(and/or ∇²n). The ingredient set per rung *is* the corpus feature schema.

**[A4]** J. P. Perdew, K. Burke, M. Ernzerhof. *Generalized Gradient Approximation Made Simple.*
Phys. Rev. Lett. **77**, 3865 (1996). DOI: 10.1103/PhysRevLett.77.3865 (erratum: Phys. Rev. Lett.
**78**, 1396 (1997), DOI: 10.1103/PhysRevLett.78.1396)
**[USE]** Primary production GGA, implemented natively in torch (`xc/gga.py`): non-empirical,
short enough to transcribe without error, and the reference functional of every pseudopotential
table used here.

**[A5]** A. D. Becke. *Density-functional exchange-energy approximation with correct asymptotic
behavior.* Phys. Rev. A **38**, 3098 (1988). DOI: 10.1103/PhysRevA.38.3098
**[USE]** B88 exchange, for BLYP and B3LYP comparison points; the second GGA implemented, to prove
the `xc/` dispatch layer is not PBE-shaped.

**[A6]** C. Lee, W. Yang, R. G. Parr. *Development of the Colle-Salvetti correlation-energy
formula into a functional of the electron density.* Phys. Rev. B **37**, 785 (1988). DOI:
10.1103/PhysRevB.37.785
**[USE]** LYP correlation. Pairs with [A5]; implemented in the Laplacian-free form of [A27].

**[A7]** S. H. Vosko, L. Wilk, M. Nusair. *Accurate spin-dependent electron liquid correlation
energies for local spin density calculations: a critical analysis.* Can. J. Phys. **58**, 1200
(1980). DOI: 10.1139/p80-159
**[USE]** VWN5 LDA correlation — one of two LDA correlation parameterisations implemented, each a
cross-check on the other.

**[A8]** J. P. Perdew, Y. Wang. *Accurate and simple analytic representation of the electron-gas
correlation energy.* Phys. Rev. B **45**, 13244 (1992). DOI: 10.1103/PhysRevB.45.13244 (APS lists
an erratum: Phys. Rev. B **98**, 079904, DOI: 10.1103/PhysRevB.98.079904)
**[USE]** PW92, the default LDA correlation for the project; its uniform-gas limit is the target
of physics gate **G1.1**.

**[A9]** J. P. Perdew, A. Zunger. *Self-interaction correction to density-functional
approximations for many-electron systems.* Phys. Rev. B **23**, 5048 (1981). DOI:
10.1103/PhysRevB.23.5048
**[USE]** Two uses: the PZ81 LDA correlation parameterisation, and the definition of one-electron
self-interaction error, which gate **G4.5** deliberately measures rather than removes.

**[A10]** J. Sun, A. Ruzsinszky, J. P. Perdew. *Strongly Constrained and Appropriately Normed
Semilocal Density Functional.* Phys. Rev. Lett. **115**, 036402 (2015). arXiv:1504.03028. DOI:
10.1103/PhysRevLett.115.036402
**[USE]** The 17-exact-constraint programme a meta-GGA can satisfy; the list is transcribed into
the gate suite as **G1.x** and is the constraint set a functional trained on this corpus is scored
against.

**[A11]** J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew, J. Sun. *Accurate and Numerically
Efficient r2SCAN Meta-Generalized Gradient Approximation.* J. Phys. Chem. Lett. **11**, 8208
(2020). DOI: 10.1021/acs.jpclett.0c02405 (correction: J. Phys. Chem. Lett. **11**, 9248 (2020),
DOI: 10.1021/acs.jpclett.0c03077)
**[USE]** The production meta-GGA, chosen over SCAN because its numerical smoothness survives
coarse real-space grids where SCAN shows grid sensitivity; third functional implemented natively.

**[A12]** J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew, J. Sun. *Construction of meta-GGA
functionals through restoration of exact constraint adherence to regularized SCAN functionals.* J.
Chem. Phys. **156**, 034109 (2022). DOI: 10.1063/5.0073623. PMID: 35065548
**[USE]** States which constraints r2SCAN keeps and which it trades away, so the gate suite
asserts none that r2SCAN provably violates.

**[A13]** A. D. Becke. *Density-functional thermochemistry. III. The role of exact exchange.* J.
Chem. Phys. **98**, 5648 (1993). DOI: 10.1063/1.464913
**[USE]** B3LYP as a reference point only; exact exchange on a real-space grid is deferred to a
later increment, and this entry records the deferral as deliberate.

**[A14]** C. Adamo, V. Barone. *Toward reliable density functional methods without adjustable
parameters: The PBE0 model.* J. Chem. Phys. **110**, 6158 (1999). DOI: 10.1063/1.478522
**[USE]** As [A13]; PBE0 is the hybrid [A17] treats as the non-empirical default, so it is the
hybrid this project would add first.

**[A15]** S. Grimme. *Semiempirical hybrid density functional with perturbative second-order
correlation.* J. Chem. Phys. **124**, 034108 (2006). DOI: 10.1063/1.2148954
*Rejected for this project.* Double hybrids need an MP2-like O(N^5) step over virtual orbitals,
which a grid code produces only at ruinous cost. Recorded to close the rung-5 question explicitly.

**[A16]** J. P. Perdew, A. Ruzsinszky. *Fourteen easy lessons in density functional theory.* Int.
J. Quantum Chem. **110**, 2801 (2010). DOI: 10.1002/qua.22829
**[USE]** Compact statement of the exact conditions (scaling, Lieb–Oxford, piecewise linearity) in
a form that translates directly into unit tests.

**[A17]** É. Brémond, Á. J. Pérez-Jiménez, C. Adamo, J. C. Sancho-García. *Contemporary DFT:
learning from traditional and recent trends for the development and assessment of accurate
exchange–correlation functionals.* Phys. Chem. Chem. Phys. **28**(6), 3779–3796 (2026) (received 2
Sep 2025, published online 30 Dec 2025; page range from OpenAlex). DOI: 10.1039/D5CP03373J. Open
access (CC BY-NC 3.0). The review traces LDA → double hybrids and discusses the three "fully
machine-learned" functionals (DM21, aPBE0-ML, Skala) as the newest class. Numbers re-checked
against the article HTML on 2026-09-16 and reused in `03_METHOD.md` (Part C, gates): GMTKN55
WTMAD-2 of 3.97 kcal/mol (DM21), 3.89 kcal/mol (Skala — this is **Skala-1.0**; Skala-1.1 reports
2.8, see [C2]), 1.7 kcal/mol (DH23, "a 12-parameter double hybrid", the lowest WTMAD-2 the review
cites; it also cites the DENS24 ensemble at 1.6 kcal/mol); W4-17 TAE MAE ≈ 1 kcal/mol (Skala, 200
values); Skala-1.0's worst GMTKN55 subsets are SIE4x4 (13.6 kcal/mol) and DIE60 (8.54 kcal/mol),
against DM21's 4.92 and 11.35.
*Corrections of 2026-09-16.* The review does **not** use the term "rung 6" — it places these
functionals on the 3rd–4th rungs of Jacob's ladder; "rung 6" was this survey's label. The
bond-length contrast "0.006–0.014 Å learned vs 0.04–0.05 Å semi-local" is **withdrawn and must not
be reused**: the review's 0.012 Å (CCse21) and 0.014 Å (LMGB35) for Skala stand, its 0.006 Å for
DM21 on LMGB35 is not traceable to the Skala paper, and its "0.04–0.05 Å" for r2SCAN, B97M-V,
B3LYP, M06-2X, ωB97X-V and ωB97M-V is given as 0.004–0.005 Å in the Skala paper's own tables —
very likely a decimal slip. The paraphrase that these functionals "exist only because of large,
diverse datasets of near-exact reference results" was not re-verified; the mandate it motivates
stands on its own.
**[USE — SOURCE OF RECORD]** Fixes the project's target and its scoring conventions. SIE4x4 = 13.6
kcal/mol for the best learned functional is why gate **G4.5** measures self-interaction error
explicitly: that residual failure is the most valuable region of the training corpus.

**[A18]** E. Caldeweyher, S. Ehlert, A. Hansen, et al. *A generally applicable atomic-charge
dependent London dispersion correction.* J. Chem. Phys. **150**, 154122 (2019). DOI:
10.1063/1.5090222
**[USE]** Dispersion enters as an additive post-SCF term with its own gate (exactly separable and
switchable). Deferred to Increment 8; the interface is reserved in `contract.py`.

**[A19]** S. Grimme, J. Antony, S. Ehrlich, H. Krieg. *A consistent and accurate ab initio
parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu.* J.
Chem. Phys. **132**, 154104 (2010). DOI: 10.1063/1.3382344
**[USE]** As [A18]; D3 is simpler and is implemented first.

**[A20]** O. A. Vydrov, T. Van Voorhis. *Nonlocal van der Waals density functional: The simpler
the better.* J. Chem. Phys. **133**, 244103 (2010). DOI: 10.1063/1.3521275
**[USE]** Named by [A17] as the non-local correlation piece inside ωB97M-V, the functional used to
generate OMol25 [B6]. Relevant only if that dataset is later consumed here.

**[A21]** J. P. Perdew, R. G. Parr, M. Levy, J. L. Balduz Jr. *Density-Functional Theory for
Fractional Particle Number: Derivative Discontinuities of the Energy.* Phys. Rev. Lett. **49**,
1691 (1982). DOI: 10.1103/PhysRevLett.49.1691
**[USE — THE EXACT CONDITION THAT MAKES THE SIE PROGRAMME CHEAP]** E(N) of an open system is
piecewise linear between adjacent integers, with a derivative discontinuity at each: the exact
reference at fractional N is the straight line between the integer endpoints, so no correlated
calculation is needed. Drives the fractional-N scans (D-22) and diagnostic **D1.2**;
`01_PROJECT.md` Part C §2.

**[A22]** A. J. Cohen, P. Mori-Sánchez, W. Yang. *Insights into Current Limitations of Density
Functional Theory.* Science **321**, 792 (2008). DOI: 10.1126/science.1158722
**[USE]** Names the two errors that dominate section K: delocalization error (convex E(N), from
self-interaction) and static correlation error (from fractional spin). The organising paper of the
diagnostic suite.

**[A23]** P. Mori-Sánchez, A. J. Cohen, W. Yang. *Discontinuous Nature of the Exchange-Correlation
Functional in Strongly Correlated Systems.* Phys. Rev. Lett. **102**, 066403 (2009). DOI:
10.1103/PhysRevLett.102.066403. See also A. J. Cohen, P. Mori-Sánchez, W. Yang, *Fractional spins
and static correlation error in density functional theory*, J. Chem. Phys. **129**, 121104 (2008),
DOI: 10.1063/1.2987202.
**[USE — THE FLAT-PLANE CONDITION]** The exact E(N, M) surface over fractional charge and spin is
a set of planes; deviation from flatness is one number for both errors, computable from integer
endpoints alone — diagnostics **D1.3** and **D1.4**, the primary diagnostics of the SIE programme.

**[A24]** Y. Goshen, E. Kraisler. *Ensemble Ground State of a Many-Electron System with Fractional
Electron Number and Spin: Piecewise-Linearity and Flat-Plane Condition Generalized.* J. Phys.
Chem. Lett. **15**(9), 2337–2343 (2024). DOI: 10.1021/acs.jpclett.3c03509
**[USE]** The modern, careful statement of the combined condition — the definition D1.4 is
implemented against. It also reports a jump in the KS potential when the spin varies at constant
N.

**[A25]** N. Q. Su, C. Li, W. Yang. *Describing strong correlation with fractional-spin correction
in density functional theory.* Proc. Natl. Acad. Sci. USA **115**(39), 9678–9683 (2018). DOI:
10.1073/pnas.1807095115
**[USE]** What a fractional-spin correction (FSLOSC) buys, and its limits; the comparison point
for diagnostic D1.3.

**[A26]** C. J. Umrigar, X. Gonze. *Accurate exchange-correlation potentials and total-energy
components for the helium isoelectronic series.* Phys. Rev. A **50**(5), 3827–3837 (1994). DOI:
10.1103/PhysRevA.50.3827
**[USE]** GGA exchange potentials (B88) diverge at the nucleus; D-52 measured the same for PBE
(`v_xc ≈ −0.017/r` on a hydrogenic density), so the KS cusp at GGA level is `Z_eff ≠ Z`: G1.11
must not assert `−2Z` there, and `v_xc` is cell-averaged near nuclei like `W`. Supplements [A4]
and [A11].

**[A27]** B. Miehlich, A. Savin, H. Stoll, H. Preuss. *Results obtained with the correlation
energy density functionals of Becke and Lee, Yang and Parr.* Chem. Phys. Lett. **157**, 200
(1989). DOI: 10.1016/0009-2614(89)87234-3
**[USE]** The form of LYP [A6] without the Laplacian of the density — the form every
implementation (libxc, PySCF) uses and `cdft.xc.gga.LYPCorrelation` implements; G0.6 verifies it
against libxc to 1e-11.

**[A28]** J. P. Perdew, A. Zunger. *Self-interaction correction to density-functional
approximations for many-electron systems.* Phys. Rev. B **23**, 5048 (1981) — this is [A9], tagged
separately for a property of it that a reader of gate G0.5 needs.
**[USE]** The PZ81 parametrisation is matched to the *printed* digits, which reproduces the
published 3.2e-5 Ha per electron discontinuity between its two branches at `r_s = 1`
(`tests/test_xc.py`); G0.5 therefore excludes `|r_s − 1| < 0.01` (D-55 item 3).

**[A29]** T. Kato. *On the eigenfunctions of many-particle systems in quantum mechanics.* Commun.
Pure Appl. Math. **10**(2), 151–177 (1957). DOI: 10.1002/cpa.3160100201
**[USE]** The cusp condition `d⟨n⟩/dr = −2Z n` at a nucleus: gate **G1.11** is the thresholded
measurement, `figures.py` draws it as `−(1/2Z) d ln⟨n⟩/dr → 1` and as the `4Z` jump of `d ln n/dz`
across a nucleus (at GGA level against the oracle's own `Z_eff/Z`, [A26]).

**[A30]** M. Levy, J. P. Perdew, V. Sahni. *Exact differential equation for the density and
ionization energy of a many-particle system.* Phys. Rev. A **30**(5), 2745–2748 (1984). DOI:
10.1103/PhysRevA.30.2745
**[USE]** The asymptotic decay `n ~ exp(−2√(2I) r)` and, with [A31], `ε_HOMO = −I` for the exact
KS potential: the `±2κ` levels and local decay exponent of the figures, and diagnostic **D1.6**
(`ε_HOMO + I`).

**[A31]** C.-O. Almbladh, U. von Barth. *Exact results for the charge and spin densities,
exchange-correlation potentials, and density-functional eigenvalues.* Phys. Rev. B **31**(6),
3231–3244 (1985). DOI: 10.1103/PhysRevB.31.3231
**[USE]** The same statements from the potential side: `v_xc → −1/r`, so a system of charge `Q`
has `v_s ~ −(Q+1)/r` and the orbital `r^(Q'/κ − 1) e^(−κr)` with the tail charge `Q'` of the
functional in use. Diagnostic **D1.7** fits `α r^(−β)` and never thresholds it (D-20).

**[A32]** A. Zupan, K. Burke, M. Ernzerhof, J. P. Perdew. *Distributions and averages of electron
density parameters: Explaining the effects of gradient corrections.* J. Chem. Phys. **106**(24),
10184–10193 (1997). DOI: 10.1063/1.474101
**[USE]** The density-weighted distributions `g(s)` and `g(r_s)` of the reduced gradient `s = |∇n|
/ (2(3π²)^(1/3) n^(4/3))` and the Wigner–Seitz radius — which ingredient values a system actually
samples. The `reduced_gradient` figure; its hydrogen medians are pinned in
`tests/test_figures.py`.

**[A33]** A. Meier de Andrade, J. Kullgren, P. Broqvist. *Quantitative and qualitative performance
of density functional theory rationalized by reduced density gradient distributions.* Phys. Rev. B
**102**(7), 075115 (2020). DOI: 10.1103/PhysRevB.102.075115
**[USE]** A recent use of the [A32] distributions to explain functional performance across bulk,
surfaces and nanoparticles (Ni); cited in the `reduced_gradient` caption as the analysis it
supports.

**[A34]** A. D. Becke, K. E. Edgecombe. *A simple measure of electron localization in atomic and
molecular systems.* J. Chem. Phys. **92**(9), 5397–5403 (1990). DOI: 10.1063/1.458517
**[USE — DELIBERATELY NOT DRAWN]** The ELF is identically 1 for every Phase 1 system (one occupied
spatial orbital, so `τ = τ_W` and `D = 0`) — a figure of a constant. It returns with open shells
(I4) and more than one orbital per spin (I5).

---

## B. Benchmarks, reference data, and verification methodology

**[B1]** L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi, S. Grimme. *A look at the density
functional theory zoo with the advanced GMTKN55 database for general main group thermochemistry,
kinetics and noncovalent interactions.* Phys. Chem. Chem. Phys. **19**, 32184 (2017). DOI:
10.1039/C7CP04913G
**[USE]** The scoring metric (WTMAD-2) that [A17] quotes throughout. GMTKN55 is a Gaussian-basis
molecular set and is not run here, but its subset structure (thermochemistry, barriers,
non-covalent, self-interaction) is the template for partitioning the scenario registry in
`physics_config.py`.

**[B2]** A. Karton, S. Daon, J. M. L. Martin. *W4-11: A high-confidence benchmark dataset for
computational thermochemistry derived from first-principles W4 data.* Chem. Phys. Lett. **510**,
165 (2011). DOI: 10.1016/j.cplett.2011.05.007
**[USE]** Sub-kJ/mol reference total atomization energies for the small closed-shell molecules of
the benchmark ladder; gate **G4.6** compares against them.

**[B3]** A. Karton, N. Sylvetsky, J. M. L. Martin. *W4-17: A diverse and high-confidence dataset
of atomization energies for benchmarking high-level electronic structure methods.* J. Comput.
Chem. **38**, 2063 (2017). DOI: 10.1002/jcc.24854
**[USE]** The set on which [A17] quotes Skala-1.0's ≈1 kcal/mol MAE; Skala-1.1 reports 0.92 on the
single-reference subset and 1.23 on all 200 values [C2]. Accuracy claims made here must be stated
on the same set to be comparable.

**[B4]** K. Lejaeghere, G. Bihlmayer, T. Björkman, et al. *Reproducibility in density functional
theory calculations of solids.* Science **351**, aad3000 (2016). DOI: 10.1126/science.aad3000
**[USE]** The Δ-gauge (71 elemental crystals, PBE) and, more importantly, the discipline behind
gates **G4.3** and **G4.4**: two codes implementing the same functional are compared on a derived
observable, never on raw total energies.

**[B5]** E. Bosoni, L. Beal, M. Bercx, et al. (45 authors). *How to verify the precision of
density-functional-theory implementations via reproducible and universal workflows.* Nat. Rev.
Phys. **6**, 45–58 (2024). arXiv:2305.17274. DOI: 10.1038/s42254-023-00655-3 Agreement measures
adopted verbatim (definitions re-checked 2026-09-16):
ε(a,b) = sqrt( ⟨[E_a(V) − E_b(V)]²⟩ / sqrt(⟨[E_a − ⟨E_a⟩]²⟩⟨[E_b − ⟨E_b⟩]²⟩) ), and
ν(a,b) = 100·sqrt( Σ_Y [w_Y·(Y_a − Y_b)/((Y_a+Y_b)/2)]² ) over Y ∈ {V₀, B₀, B₁} with weights
1, 1/20, 1/400. The "excellent/good" cut-offs used here (**ε ≲ 0.06 / ≲ 0.2**, **ν ≲ 0.1 / ≲
0.33**) are stated in the paper's Supplementary Section S7. That section could not be retrieved on
2026-09-16, so the cut-offs are carried as transcribed and should be re-confirmed against the
supplement before they are quoted. Reference set: 960 equations of state (Z = 1–96), WIEN2k vs
FLEUR.
**[USE — METHODOLOGY ANCHOR]** Gate **G4.4** applies the ε construction to a molecular
dissociation curve instead of an equation of state; also the source of the rule that absolute
total energies are never comparable across codes.

**[B6]** D. S. Levine, M. Shuaibi, et al. (Meta FAIR; 23–24 authors depending on version). *The
Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models.* arXiv:2505.08762 (2025; v2 March
2026; no journal version found). ωB97M-V/def2-TZVPD single points (>100M per the paper body; one
reading of the current abstract gives >140M — the cited version must be stated), 83 elements,
systems to 350 atoms, ~83M unique systems, 6.6 billion CPU core-hours. The fair-chem documentation
states that the electronic-structure data (densities, Fock matrices) were generated but are not
yet publicly released.
**[USE]** Two roles: the scale reference that says what "a dataset worth training on" means, and a
source of reference geometries, so the sampling here is not arbitrary. Cited by [A17] as the
substrate of the newest learned functionals.

**[B7]** S. Song, S. Vuckovic, E. Sim, K. Burke. *Density-Corrected DFT Explained: Questions and
Answers.* J. Chem. Theory Comput. **18**, 817 (2022). DOI: 10.1021/acs.jctc.1c01045
**[USE]** The functional-error / density-error decomposition — decisive for the data schema: every
HDF5 record stores the converged density alongside its energies, because a corpus of energies
alone teaches a model the sum of two different errors.

**[B8]** S. Kim, D.-G. Lee, G. Kim, Y. Kim, M. Sogal, S. Crisostomo, K. Burke, E. Sim. *Analyzing
density-driven errors: Principles and pitfalls.* J. Chem. Phys. **164**, 064106 (2026). DOI:
10.1063/5.0303445. PMID: 41665441
**[USE]** Current practice for measuring the density-driven error correctly (pitfalls: inaccurate
interpolators, proxy densities, conflating error measures); defines what a record must contain to
make it computable post hoc.

**[B9]** L. Goerigk, S. Grimme. *A thorough benchmark of density functional methods for general
main group thermochemistry, kinetics, and noncovalent interactions.* Phys. Chem. Chem. Phys.
**13**, 6670 (2011). DOI: 10.1039/c0cp02984j
**[USE]** Predecessor of [B1]; supplies the historical error baselines [A17] compares against.

**[B10]** S. Kotochigova, Z. H. Levine, E. L. Shirley, M. D. Stiles, C. W. Clark. *Atomic
Reference Data for Electronic Structure Calculations* (LSD and LDA total energies and eigenvalues,
H–U). NIST Standard Reference Database 141, DOI: 10.18434/T4ZP4F; underlying paper:
*Local-density-functional calculations of the energy of atoms*, Phys. Rev. A **55**, 191 (1997),
DOI: 10.1103/PhysRevA.55.191 (erratum: Phys. Rev. A **56**, 5191 (1997)).
**[USE]** Free-atom LDA total energies and eigenvalues at all-electron level — the only absolute
numbers in the project that can be compared directly, and only on the all-electron path; gate
**G4.7**.

**[B11]** M. G. Medvedev, I. S. Bushmarinov, J. Sun, J. P. Perdew, K. A. Lyssenko. *Density
functional theory is straying from the path toward the exact functional.* Science **355**(6320),
49–52 (2017). DOI: 10.1126/science.aah5975; PMID 28059761
**[USE]** Judges densities, not energies, by three quantities — n, |∇n| and ∇²n — against CCSD
references. The `density_error` figure measures the same three deviations against a reference *of
the same functional*, plus `∫|n − n_ref|`: solver error, not the functional error of the paper
(that needs [B7]).

---

## C. Machine-learned exchange–correlation functionals (the downstream consumer)

**[C1]** J. Kirkpatrick, B. McMorrow, D. H. P. Turban, A. L. Gaunt, J. S. Spencer, A. G. D. G.
Matthews, et al. (17 authors). *Pushing the frontiers of density functionals by solving the
fractional electron problem (DM21).* Science **374**, 1385–1389 (2021). DOI:
10.1126/science.abj6511 A local range-separated hybrid trained with explicit fractional-charge and
fractional-spin constraints: 1161 molecular reactions plus 1074 fractional-charge/spin densities
for H–Ar. (The WTMAD-2 of 3.97 kcal/mol is quoted by [C2] and [A17]; the manuscript itself reports
the GMTKN55 "mean of means", 1.50 kcal/mol.)
**[USE — DEFINES THE TARGET]** Two consequences for the baseline: occupation numbers are floats
from day one, so fractional electron numbers and spins can be run; and the training signal that
mattered — dissociation curves of H₂⁺, H₂ and N₂ — is exactly what the benchmark ladder must
produce.

**[C2]** G. Luise, C.-W. Huang, T. Vogels, D. P. Kooi, S. Ehlert, S. Lanius, et al. (Microsoft
Research AI for Science; 25–31 authors depending on version). *Accurate and scalable
exchange-correlation with deep learning (Skala).* arXiv:2506.14665 (v1 17 Jun 2025 … v6 21 Apr
2026; no journal version found as of 2026-09-16). MIT licence; `pip install skala`; ~385k
parameters.
*Versions.* **Skala-1.0** (the functional of the first preprint, v1 of 17 Jun 2025) has GMTKN55
WTMAD-2 **3.89** kcal/mol — the value [A17] quotes — trained on ~150k reaction energies including
~80k MSR-ACC/TAE atomization energies. **Skala-1.1** (v6) reports **2.8** kcal/mol (ωB97M-V:
3.23), W4-17 MAE **0.92** kcal/mol on the single-reference subset (183/200) and **1.23** overall,
trained on ~400k energy differences (MSR-ACC, W1-F12/W1w, [C31]) plus ~80k public data. Cost
within ~30 % of r2SCAN on GPU. Runs in PySCF, GPU4PySCF, ASE and GauXC; in CP2K since 2026 [C21].
Stated limitations: fixed D3 dispersion; strongly correlated / multireference data "remains an
obstacle".
**[USE]** The strongest argument for this architecture: Skala is *semi-local in cost* — it learns
non-local representations from semilocal grid features and reaches hybrid-level accuracy without
any exact-exchange machinery, so it plugs into a real-space grid code unchanged.

**[C3]** L. Li, S. Hoyer, R. Pederson, R. Sun, E. D. Cubuk, P. Riley, K. Burke. *Kohn-Sham
Equations as Regularizer: Building Prior Knowledge into Machine-Learned Physics.* Phys. Rev. Lett.
**126**, 036401 (2021). arXiv:2009.08551. DOI: 10.1103/PhysRevLett.126.036401 Generalisation comes
from backpropagating through the entire SCF loop rather than fitting E_xc pointwise (1D model
systems, DMRG references; two H₂ separations suffice for the whole 1D H₂ curve).
**[USE — ARCHITECTURAL CONSTRAINT, narrowed by D-47]** What this repository owes is autograd
*inside* the functional and a pure, differentiable one-step fixed-point map (`SCFStepProtocol`,
contract 1.5.0, D-49); training through the SCF belongs to the downstream project, by implicit
differentiation [C5].

**[C4]** M. F. Kasim, S. M. Vinko. *Learning the Exchange-Correlation Functional from Nature with
Fully Differentiable Density Functional Theory.* Phys. Rev. Lett. **127**, 126403 (2021).
arXiv:2102.04229. DOI: 10.1103/PhysRevLett.127.126403
**[USE]** The same idea against experimental data (eight diatomic data points; tested on 104 G2
atomization energies); establishes the gradient path E_expt → E_total → SCF → E_xc[θ] that the
frozen interfaces of `contract.py` must keep unbroken.

**[C5]** M. F. Kasim, S. Lehtola, S. M. Vinko. *DQC: A Python program package for differentiable
quantum chemistry.* J. Chem. Phys. **156**, 084801 (2022). DOI: 10.1063/5.0076202
**[USE]** The closest existing implementation of what this project builds, in PyTorch; its
implicit differentiation through the SCF fixed point is adopted here rather than unrolling the
loop, which would blow the 6 GB memory budget.

**[C6]** P. A. M. Casares, J. S. Baker, M. Medvidović, R. dos Reis, J. M. Arrazola. *GradDFT. A
software library for machine learning enhanced density functional theory.* J. Chem. Phys. **160**,
062501 (2024). arXiv:2309.15127. DOI: 10.1063/5.0181037
**[USE]** JAX equivalent of [C5]; its functional-parameterisation API (a learned functional as
coefficients over classical ingredients) is the interface shape the `xc/` dispatch layer copies,
so a Skala- or GradDFT-style functional drops in without touching the SCF.

**[C7]** M. Bogojeski, L. Vogt-Maranto, M. E. Tuckerman, K.-R. Müller, K. Burke. *Quantum chemical
accuracy from density functional approximations via machine learning.* Nat. Commun. **11**, 5223
(2020). DOI: 10.1038/s41467-020-19093-1
**[USE]** Learns the map DFT density → CCSD(T) energy; the reason every record stores the full
converged density rather than summary statistics.

**[C8]** R. Nagai, R. Akashi, O. Sugino. *Completing density functional theory by machine learning
hidden messages from molecules.* npj Comput. Mater. **6**, 43 (2020). DOI:
10.1038/s41524-020-0310-0 A neural functional (LSDA/GGA/meta-GGA inputs plus a non-local "near
region approximation") trained on three molecules only (H₂O, NH₃, NO; atomization energies and
CCSD densities) that still generalises to hundreds of molecules.
**[USE]** Evidence that the corpus does not need OMol25 scale to be useful: a few hundred
well-converged, gate-passing systems with full densities may be worth more than millions of
energies.

**[C8b]** S. Dick, M. Fernández-Serra. *Machine learning accurate exchange and correlation
functionals of the electronic density (NeuralXC).* Nat. Commun. **11**, 3509 (2020). DOI:
10.1038/s41467-020-17265-7
**[USE]** Atom-centred density descriptors on top of PBE trained towards CCSD(T); a second data
point for "small, high-quality corpora transfer".

**[C9]** J. Wu, S.-M. Pun, X. Zheng, G. Chen. *Construct exchange-correlation functional via
machine learning.* J. Chem. Phys. **159**, 090901 (2023) (Perspective). DOI: 10.1063/5.0150587
**[USE]** Survey of the design space (pointwise vs global, energy vs potential targets, constraint
enforcement); used to structure the feature list in the HDF5 schema.

**[C10]** B. Kalita, R. Pederson, J. Chen, L. Li, K. Burke. *How Well Does Kohn–Sham Regularizer
Work for Weakly Correlated Systems?* J. Phys. Chem. Lett. **13**(11), 2540–2547 (2022). DOI:
10.1021/acs.jpclett.2c00371. PMID: 35285630
**[USE]** Where the KSR approach of [C3] fails and how far a spin-adapted version gets (non-local
variant: 2.7 mHa MAE on 1D test molecules); read as a risk register for the downstream project.

**[C11]** X. Zhang (Peking University). *End-to-End Differentiable Learning of a Single Functional
for DFT and Linear-Response TDDFT.* arXiv:2602.05345 (2026).
**[USE]** One functional trained against ground-state and response data — direct evidence for
keeping a clean `Hamiltonian.apply` seam so a TDDFT propagator can be added without
re-architecting.

**[C12]** A. von Strachwitz, K. K. Alaa El-Din, A. C. C. Dutra, S. M. Vinko. *Data-efficient
learning of exchange-correlation functionals with differentiable DFT.* Mach. Learn.: Sci. Technol.
**7**(2), 025001 (2026). DOI: 10.1088/2632-2153/ae3c5a
**[USE]** How much data a differentiable-DFT training run needs (recovers known LDAs from small
synthetic datasets; strongly dependent on data amount, type and initialisation). Informs the size
of the I9 corpus qualitatively — the paper gives trends, not a transferable target size.

**[C13]** Y. Zhuang, Y. Gu, B. Zhang, J. Wu, G. Chen. *Machine Learning Accurate
Exchange–Correlation Potentials for Reducing Delocalization Error in Density Functional Theory.*
JACS Au **5**(8), 4002–4010 (2025). DOI: 10.1021/jacsau.5c00632
**[USE]** Learns v_xc(r) rather than E_xc, which requires inverted reference potentials — the
reason Increment 10 (KS inversion, section G) is core rather than optional.

**[C14]** The DM21 controversy. (a) Comment: I. S. Gerasimov, T. V. Losev, E. Yu. Epifanov, I.
Rudenko, I. S. Bushmarinov, A. A. Ryabov, P. A. Zhilyaev, M. G. Medvedev, *Comment on "Pushing the
frontiers of density functionals by solving the fractional electron problem"*, Science **377**,
eabq3385 (2022), DOI: 10.1126/science.abq3385. (b) Reply: J. Kirkpatrick et al., *Response to
Comment on …*, Science **377**, eabq4282 (2022), DOI: 10.1126/science.abq4282. (c) News coverage:
*Scientists doubt that DeepMind's AI is as good for fractional-charge systems as it seems*,
Phys.org (Skoltech press release), August 2022.
**[USE]** A caution: DM21's generalisation claim was contested (the comment alleges ~50 % overlap
between training data and the bond-breaking benchmark; the authors dispute it). Reinforces the
rule that every generated record carries its gate report, so downstream claims are auditable.

**[C15]** R. Nagai, R. Akashi, O. Sugino. *Machine-learning-based exchange correlation functional
with physical asymptotic constraints.* Phys. Rev. Research **4**, 013106 (2022). DOI:
10.1103/PhysRevResearch.4.013106
**[USE]** How to impose asymptotic constraints on a neural functional by construction; informs
which constraint diagnostics the corpus must make computable.

**[C16]** R. Remme, T. Kaczun, M. Scheurer, A. Dreuw, F. A. Hamprecht. *KineticNet: Deep learning
a transferable kinetic energy functional for orbital-free density functional theory.* J. Chem.
Phys. **159**, 144113 (2023). DOI: 10.1063/5.0158275
*Out of scope* (orbital-free DFT is outside the mandate), retained because it shares the feature
pipeline: if T_s[n] is ever targeted, the same HDF5 records serve, provided τ(r) is stored — which
it is.

**[C17]** S. Manzhos, J. Lüder, P. Golub, M. Ihara. *Machine learning-guided construction of an
analytic kinetic energy functional for orbital free density functional theory.* Mach. Learn.: Sci.
Technol. **6**(3), 035002 (2025). arXiv:2502.05411. DOI: 10.1088/2632-2153/ade7ca
*Out of scope*, same reasoning as [C16].

**[C18]** M. Chen, M. Pavanello, W. Mi, M. Ihara, S. Manzhos. *Machine Learning-Enhanced
Orbital-Free Density Functional Theory* (review). J. Chem. Theory Comput. **22**(7), 3127–3143
(2026). DOI: 10.1021/acs.jctc.5c01995. PMID: 41889092
*Out of scope*, same reasoning as [C16].

**[C19]** B. Kanungo, J. Hatch, P. M. Zimmerman, V. Gavini. *Learning local and semi-local density
functionals from exact exchange–correlation potentials and energies.* Sci. Adv. **11**(38),
eady8962 (2025). DOI: 10.1126/sciadv.ady8962. Preprint arXiv:2409.06498. Neural LDA and GGA
functionals trained on exact XC potentials obtained by inverse DFT on configuration-interaction
densities of **five atoms and two molecules** (Li, C, N, O, Ne; H₂, LiH), reaching SCAN-level
accuracy on unseen molecules: NNGGA 1.9 kcal/mol per atom MAE in total energies on 97 G2
molecules, with atomization energies and BH76-subset barrier heights on par with SCAN. Stated
limitations: the NNGGA violates the UEG limit and may not converge in solids.
**[USE — THE TEMPLATE FOR A SINGLE-GROUP CONTRIBUTION]** The strongest evidence that
potential-target learning is the high-leverage route for a group without industrial compute, and
the direct argument for promoting the KS-inversion increment from optional to core (D-17).

**[C20]** M. S. Abdallah, Z. Jin, B. Kozinsky, K. Bystrom. *Machine-learned exchange–correlation
functionals for molecules, solids, and reactive surfaces* (CIDER26SS). arXiv:2608.21525 (2026).
Harvard / MIT / **Bosch Research** / Flatiron Institute. Semilocal ingredients augmented with
non-local integral features via Gaussian-process regression, trained on all of GMTKN55 plus the
transition-metal and total-atomic-energy parts of GSCDB137 [C30] and a bespoke EXX+RPA dataset for
bulk transition metals and surfaces. GMTKN55 WTMAD-2 = 4.10 kcal/mol at fixed training densities
(4.12 self-consistently); resolves the CO/Pt(111) site puzzle with an atop adsorption energy of
−1.23 eV against an experimental −1.28 to −1.20 eV, with the Pt lattice constant 3.931 vs 3.924 Å.
Stated limitations: no r⁻⁶ dispersion tail, no insulator/semiconductor training data.
**[USE]** Current state of the art for *coverage*; and an industrial R&D lab co-authoring
functional development is itself the evidence for `01_PROJECT.md` Part D, Part III.

**[C21]** F. Pöschel, J. Pototschnig, F. Stein, A. Knüpfer, T. Vogels, S. Battaglia, S. Ehlert, J.
Hutter, T. D. Kühne. *Molecular Implementation of the Machine-Learned Skala Exchange–Correlation
Functional in CP2K through GauXC.* arXiv:2608.19033 (2026).
**[USE]** Skala [C2] went from first preprint (June 2025) to a production code (CP2K via GauXC, as
Skala-1.1, molecular; dietGMTKN55 MAD 1.255 vs the 1.235 kcal/mol reference) in roughly fourteen
months, itself reported in a preprint: the deliverable that reaches industry is the *functional*,
not the solver.

**[C22]** E. S. Eberhard, V. Kotsev, T. Güthle, S. Günnemann (TU Munich). *Transferable
SCF-Acceleration through Solver-Aligned Initialization Learning* (SAIL). arXiv:2604.21657 (2026;
v2 8 May 2026). Backpropagates through the SCF to train an initial guess on *solver dynamics*
rather than on ground-state targets. The reported **37 % (PBE), 33 % (SCAN), 28 % (B3LYP)** are
reductions of **ERIC** — an effective relative iteration count based on Fock builds, introduced in
the paper — on QM40 (molecules up to 4× larger than the QM9 training set); the **1.35× wall-time**
gain at the hybrid level is on QMugs (10× larger molecules). Gaussian-basis (def2-SVP), timings
with GPU4PySCF; the element set is not stated.
**[USE — THE SAFE ML WIN]** It changes only the *path* to the fixed point, so gate **G3.4**
verifies it outright. The real-space grid analogue is open, with [D4] as the nearest precedent.

**[C23]** (a) Z. Y. Yescas-Ramos, A. Álvarez-García, H. E. Sauceda, *Towards Accelerated SCF
Workflows with Equivariant Density-Matrix Learning and Analytic Refinement*, arXiv:2604.27256
(2026) (49–81 % fewer SCF iterations on small molecules); (b) S. Hazra, U. Patil, S. Sanvito,
*Predicting the One-Particle Density Matrix with Machine Learning*, J. Chem. Theory Comput.
**20**(11), 4569–4578 (2024), DOI: 10.1021/acs.jctc.4c00042.
**[USE]** The density-matrix flavour of the same idea as [C22].

**[C24]** L. Zhang, G. Duan, D. Luo. *WF-Bench: A Benchmark for Neural Network WaveFunction
Expressivity and Scaling Laws.* arXiv:2605.29683 (2026) (the PDF footer indicates ICML 2026; not
independently confirmed).
**[USE]** Neural-network wavefunctions are the most credible route to reference data above CCSD(T)
where CCSD(T) is out of reach — stretched bonds, fractional charges and spins, the
self-interaction region. A future label source for [C19]-style training, not a method this project
implements.

**[C25]** I. Creed, T. Rein, I. Vitenburgs, W. G. Stark, V. Ellingsson, A. Y. Ismail, et al. (24
authors; corresponding author K. T. Butler). *Six Open Questions in Machine-Learned Interatomic
Potential Foundation Models.* arXiv:2606.07327 (2026). The six questions: minimal definition of an
atomistic foundation model; more data, better data or better models; long-range interactions;
discovery of new physics; scaling to useful simulations; how to know whether an MLIP is any good.
Verified in the paper: mixed PBE/PBE+U training data introduce systematic and unsystematic noise;
citing earlier work, it notes that a 90:10 PBE:SCAN mixture can match a model trained on eight
times more pure SCAN data.
**[USE]** Read before assuming the MD route is open ground. *This project's reading (not a claim
of the paper):* such models take over the throughput work, and what they cannot fix is a
systematically wrong underlying functional, which they inherit.

**[C26]** K. Bystrom, B. Kozinsky. *CIDER: An Expressive, Nonlocal Feature Set for Machine
Learning Density Functionals with Exact Constraints.* J. Chem. Theory Comput. **18**(4), 2180–2192
(2022). DOI: 10.1021/acs.jctc.1c00904. arXiv:2109.02788
**[USE — THE FEATURE CLASS TO TARGET]** Non-local features *of the density* — not exact exchange —
built to satisfy uniform scaling by construction: the middle path between semi-local ingredients
(which provably cannot fix one-electron self-interaction) and exact exchange (ruinous on a grid).

**[C27]** K. Bystrom, B. Kozinsky. *Nonlocal machine-learned exchange functional for molecules and
solids.* Phys. Rev. B **110**, 075130 (2024). arXiv:2303.00682. DOI: 10.1103/PhysRevB.110.075130
The feature form is explicit: G_i(r₁) ∝ ∫ d³r₂ Φ(a(r₂), b_i(r₁), r₁₂) n(r₂) with a Gaussian kernel
Φ(a,b,r) = exp[−(a+b)r²] and a density- and τ-dependent length scale a[n](r), made tractable by
expanding the kernel in fixed exponents so the integral becomes a convolution. The atom-centred
(Gaussian-basis) implementation is bottlenecked by a spline evaluation quadratic in system size;
the plane-wave/PAW implementation is **Θ(N log N)** via FFT convolutions. Exact exchange is O(N⁴).
(Only a few SCF convergence problems across 2462 molecular and 453 periodic systems.)
**[USE — THE ARCHITECTURAL ARGUMENT FOR A GRID]** A uniform real-space grid admits the same FFT
route, so it is expected — this project's inference, not the paper's — to be among the cheapest
substrates for this non-local feature class.

**[C28]** A. Bajaj, J. P. Janet, H. J. Kulik. *Communication: Recovering the flat-plane condition
in electronic structure theory at semi-local DFT cost.* J. Chem. Phys. **147**, 191101 (2017).
arXiv:1710.02378. DOI: 10.1063/1.5008981
**[USE]** Proof of existence for the target: the flat-plane condition [A23] is recoverable at
semi-local *cost* (projectors — semi-local in cost, not in ingredients). With [C26]/[C27] this
fixes the design space: fix SIE with non-local information that is still cheap, not with exact
exchange.

**[C29]** M. Medvidović, J. C. Umana, I. Ahmadabadi, D. Di Sante, J. Flick, A. Rubio. *Neural
network distillation of orbital dependent density functional theory.* Phys. Rev. Research **7**,
023113 (2025). arXiv:2410.16408. DOI: 10.1103/PhysRevResearch.7.023113
**[USE]** Compresses the orbital-dependent meta-GGAs r2SCAN and TPSS into τ-free neural "global
density approximations"; a third route into the same design space.

**[C30]** J. Liang, M. Head-Gordon. *Gold-Standard Chemical Database 137 (GSCDB137): A Diverse Set
of Accurate Energy Differences for Assessing and Developing Density Functionals.* J. Chem. Theory
Comput. **21**(24), 12601–12621 (2025). arXiv:2508.13468. DOI: 10.1021/acs.jctc.5c01380 137 data
sets / 8377 entries updating GMTKN55 and MGCDB84 (W4-17 references, spin-contaminated points
removed, transition-metal chemistry and molecular properties added); part of the CIDER26SS
training data.
**[USE]** The natural successor benchmark to [B1] for any functional trained on this corpus.

**[C31]** S. Ehlert, J. Hermann, T. Vogels, V. Garcia Satorras, S. Lanius, M. Segler, et al. (14
authors). *Accurate Chemistry Collection: Coupled cluster atomization energies for broad chemical
space.* Sci. Data **13**, 951 (2026). arXiv:2506.14492. DOI: 10.1038/s41597-026-07200-8
**[USE]** MSR-ACC/TAE25: 73,040 CCSD(T)/CBS (W1-F12) total atomization energies of closed-shell
neutral molecules with ≤5 non-hydrogen atoms (elements up to Ar), CDLA-Permissive-2.0 — the
published part of the label set behind Skala [C2]. The Zenodo record title changed from "77k" (v2)
to "73k" (v3); cite the version.

**[C32]** D. Khan, A. J. A. Price, B. Huang, M. L. Ach, O. A. von Lilienfeld. *Adapting hybrid
density functionals with machine learning.* Sci. Adv. **11**(5), eadt7769 (2025).
arXiv:2402.14793. DOI: 10.1126/sciadv.adt7769
**[USE]** aPBE0 / aPBE0-ML, the third learned functional discussed by [A17]: a per-system
exact-exchange fraction predicted by kernel ridge regression before the SCF (QM9
atomization-energy MAE 4.68 → 1.32 kcal/mol vs PBE0). Stated limitation: near-equilibrium only.

---

## D. Real-space grid discretisation and reference implementations

**[D1]** J. R. Chelikowsky, N. Troullier, Y. Saad. *Finite-difference-pseudopotential method:
Electronic structure calculations without a basis.* Phys. Rev. Lett. **72**, 1240 (1994). DOI:
10.1103/PhysRevLett.72.1240
**[USE — FOUNDATION OF THE DISCRETISATION]** The method the solver implements: uniform Cartesian
grid, high-order finite-difference Laplacian, Dirichlet boundary on a box, no basis set and
therefore no basis-set superposition error to contaminate the energy differences of the corpus.

**[D2]** J. R. Chelikowsky, N. Troullier, K. Wu, Y. Saad. *Higher-order finite-difference
pseudopotential method: An application to diatomic molecules.* Phys. Rev. B **50**, 11355 (1994).
DOI: 10.1103/PhysRevB.50.11355
**[USE]** The order-2p finite-difference coefficients and their convergence behaviour; sets the
stencil order (2p = 8 by default, 12 available) and gate **G0.1**, which asserts that the measured
convergence order matches the nominal one.

**[D3]** L. Kronik, A. Makmal, M. L. Tiago, et al. *PARSEC — the pseudopotential algorithm for
real-space electronic structure calculations: recent advances and novel applications to
nano-structures.* Phys. Status Solidi B **243**, 1063 (2006). DOI: 10.1002/pssb.200541463
**[USE]** Reference architecture for a production real-space molecular code; the module
decomposition is taken over, but with a box rather than a sphere (simpler, FFT-compatible).

**[D4]** Z. Zhang, C. Mora Perez, P. Kwon, M. Head-Gordon, J. Qian. *PARSEC.py: A Python-Based
Real-Space Kohn–Sham Density Functional Theory Code Accelerated by Machine Learned Charge
Density.* J. Comput. Chem. **47**(23), e70482 (2026). DOI: 10.1002/jcc.70482. PMID: 42627649 *(the
abstract gives no numerical speed-ups)*
**[USE — CLOSEST PRIOR ART]** A Python real-space KS code that uses a machine-learned charge
density to accelerate the SCF; read for its treatment of the Python/compute-kernel boundary.

**[D5]** N. Tancogne-Dejean, M. J. T. Oliveira, X. Andrade, et al. *Octopus, a computational
framework for exploring light-driven phenomena and quantum dynamics in extended and finite
systems.* J. Chem. Phys. **152**, 124119 (2020). arXiv:1912.07921. DOI: 10.1063/1.5142502
**[USE — PRIMARY CROSS-CHECK CODE]** Real-space, finite-difference, ONCV pseudopotentials,
molecules in a box — the same discretisation, so the only external code directly comparable to
this solver, and the cross-validation oracle of gates **G4.3** and **G4.4**. Also the reference for
default spacings and box radii.

**[D6]** X. Andrade, D. Strubbe, U. De Giovannini, et al. *Real-space grids and the Octopus code
as tools for the development of new simulation approaches for electronic systems.* Phys. Chem.
Chem. Phys. **17**, 31371 (2015). DOI: 10.1039/C5CP00351B
**[USE]** The design-rationale paper for real-space grids — why they parallelise, why they are
basis-set-free, where their errors come from; its grid-convergence section underpins gate
**G3.1**.

**[D7]** J. J. Mortensen, A. H. Larsen, M. Kuisma, et al. *GPAW: An open Python package for
electronic structure calculations.* J. Chem. Phys. **160**, 092503 (2024). arXiv:2310.14776. DOI:
10.1063/5.0182685
**[USE]** Second cross-check code and the model for a Python-first architecture with compute
kernels below. PAW rather than norm-conserving, so comparison is on energy *differences* only.

**[D8]** J. Enkovaara, C. Rostgaard, J. J. Mortensen, et al. *Electronic structure calculations
with GPAW: a real-space implementation of the projector augmented-wave method.* J. Phys.: Condens.
Matter **22**, 253202 (2010). DOI: 10.1088/0953-8984/22/25/253202
**[USE]** The real-space PAW implementation details, including multigrid Poisson and the
coarse/fine grid split.

**[D9]** Q. Xu, A. Sharma, B. M. Comer, H. Huang, E. Chow, A. J. Medford, J. E. Pask, P.
Suryanarayana. *SPARC: Simulation Package for Ab-initio Real-space Calculations.* SoftwareX
**15**, 100709 (2021). DOI: 10.1016/j.softx.2021.100709. Newer program paper: B. Zhang et al.,
*SPARC v2.0.0: Spin-orbit coupling, dispersion interactions, and advanced exchange–correlation
functionals*, Software Impacts **20**, 100649 (2024), DOI: 10.1016/j.simpa.2024.100649 (record
from OpenAlex).
**[USE]** A modern, compact real-space FD code; its published convergence tables (energy and force
vs grid spacing) give realistic expectations for the **G3.1** thresholds.

**[D10]** Q. Xu, A. Sharma, P. Suryanarayana. *M-SPARC: Matlab-Simulation Package for Ab-initio
Real-space Calculations.* SoftwareX **11**, 100423 (2020). arXiv:1912.08903. DOI:
10.1016/j.softx.2020.100423
**[USE]** A readable implementation of the same algorithms; the algorithmic cross-reference when a
formula in a paper is ambiguous.

**[D11]** P. Motamarri, S. Das, S. Rudraraju, K. Ghosh, D. Davydov, V. Gavini. *DFT-FE — A
massively parallel adaptive finite-element code for large-scale density functional theory
calculations.* Comput. Phys. Commun. **246**, 106853 (2020). arXiv:1903.10959. DOI:
10.1016/j.cpc.2019.07.016
**[USE]** Finite-element real-space DFT: not this project's discretisation, but its Chebyshev
filtering at scale and its mixed-precision strategy transfer directly.

**[D12]** S. Das, P. Motamarri, V. Subramanian, et al. *DFT-FE 1.0: A massively parallel hybrid
CPU-GPU density functional theory code using finite-element discretization.* Comput. Phys. Commun.
**280**, 108473 (2022). arXiv:2203.07820. DOI: 10.1016/j.cpc.2022.108473
**[USE]** The GPU port; read specifically for which parts of the SCF they keep in double
precision.

**[D13]** N. Kodali, G. Panigrahi, N. Gupta, K. Ramakrishnan, Sundaresan G., R. Panch, et al. (9
authors; corresponding author P. Motamarri). *Towards exascale fully relativistic pseudopotential
density functional theory calculations enabled by mixed-precision computation and
compressed-communication using residual based subspace iteration.* arXiv:2605.30128 (2026).
**[USE — PRECISION POLICY EVIDENCE]** State of the art on which SCF operations tolerate reduced
precision and which do not (residual-based Chebyshev filtering tolerant to inexact products,
FP32/TF32, compressed MPI; up to 9,600 GPUs on Aurora) — the empirical basis for the precision
policy of `03_METHOD.md` Part A.

**[D14]** L. Genovese, T. Deutsch, A. Neelov, S. Goedecker, G. Beylkin. *Efficient solution of
Poisson's equation with free boundary conditions.* J. Chem. Phys. **125**, 074105 (2006). DOI:
10.1063/1.2335442
**[USE]** The interpolating-scaling-function Poisson solver: O(N log N), exact free boundary
conditions, no artificial periodicity. Primary alternative in `operators/poisson.py`.

**[D15]** L. Genovese, A. Neelov, S. Goedecker, et al. *Daubechies wavelets as a basis set for
density functional pseudopotential calculations.* J. Chem. Phys. **129**, 014109 (2008). DOI:
10.1063/1.2949547
*Rejected as a discretisation* (wavelets add machinery that is not needed at 100 atoms), but the
paper's Poisson and pseudopotential treatments are reused.

**[D16]** T. Ono, K. Hirose. *Real-space electronic-structure calculations with a time-saving
double-grid technique.* Phys. Rev. B **72**, 085115 (2005). DOI: 10.1103/PhysRevB.72.085115
(arXiv:cond-mat/0412571 carries the title "…with timesaving double-grid technique"). Original
method: T. Ono, K. Hirose, *Timesaving Double-Grid Method for Real-Space Electronic-Structure
Calculations*, Phys. Rev. Lett. **82**, 5016 (1999), DOI: 10.1103/PhysRevLett.82.5016.
**[USE]** The double-grid technique that removes aliasing when sharp pseudopotential terms are
projected onto a coarse grid; required for accurate forces on the pseudopotential path (I5).

**[D17]** D. Roller, A. M. Rappe, L. Kronik, O. Hellman. *Finite Difference Interpolation for
Reduction of Grid-Related Errors in Real-Space Pseudopotential Density Functional Theory.* J.
Chem. Theory Comput. **19**(13), 3889–3899 (2023). DOI: 10.1021/acs.jctc.3c00217
**[USE — EGG-BOX CONTROL]** The egg-box error — a spurious dependence of the energy on where atoms
sit relative to grid points — is the most dangerous silent failure of a real-space code: smooth,
small and unphysical. Gate **G2.7** is built from this paper; **G2.6** (forces vs finite
differences) cites it.

**[D18]** The mask-function / Fourier-filtering technique, in three papers:
(i) R. D. King-Smith, M. C. Payne, J. S. Lin, *Real-space implementation of nonlocal
pseudopotentials for first-principles total-energy calculations*, Phys. Rev. B **44**, 13063
(1991), DOI: 10.1103/PhysRevB.44.13063;
(ii) L.-W. Wang, *Mask-function real-space implementations of nonlocal pseudopotentials*, Phys.
Rev. B **64**, 201107(R) (2001), DOI: 10.1103/PhysRevB.64.201107;
(iii) M. Tafipolsky, R. Schmid, *A general and efficient pseudopotential Fourier filtering scheme
for real space methods using mask functions*, J. Chem. Phys. **124**, 174102 (2006), DOI:
10.1063/1.2193514.
*(Corrected 2026-09-16: the earlier entry cited a non-existent "Fattebert, King-Smith, Vanderbilt,
Phys. Rev. B 62, 1713"; PRB 62, 1713 is J.-L. Fattebert, J. Bernholc, DOI:
10.1103/PhysRevB.62.1713.)*
**[USE]** Fourier filtering of pseudopotential projectors removes components the grid cannot
represent; complementary to [D16], and both are needed for egg-box control.

**[D19]** T. Ono, M. Heide, N. Atodiresei, P. F. Baumeister, S. Tsukamoto, S. Blügel. *Real-space
electronic structure calculations with full-potential all-electron precision for transition
metals.* Phys. Rev. B **82**, 205115 (2010). arXiv:1009.0800. DOI: 10.1103/PhysRevB.82.205115
**[USE]** Evidence on how far real-space grids can be pushed toward all-electron accuracy (a
real-space FD PAW scheme matching plane-wave PAW and FLAPW). Bounds the claims made here.

**[D20]** M. Dogan, K.-H. Liou, J. R. Chelikowsky. *Solving the electronic structure problem for
over 100 000 atoms in real space.* Phys. Rev. Materials **7**, L063001 (2023). arXiv:2303.00790.
DOI: 10.1103/PhysRevMaterials.7.L063001 (PARSEC, Si nanocluster Si₁₀₇₆₄₁H₉₀₈₄)
**[USE]** Scaling reference; confirms the real-space approach is not a toy at the 50–100 atom
target.

**[D21]** R. Kumar, D. Codony, P. Suryanarayana, A. Sharma. *Unified open-boundary electrostatics
in real-space density functional theory.* arXiv:2608.08474 (2026).
**[USE]** A recent method paper (implemented in SPARC) on open-boundary Hartree terms; used to
choose between the interpolating-scaling-function solver [D14] and the reciprocal-space treatments
[F14], [F15].

**[D22]** S. R. Jensen, S. Saha, J. A. Flores-Livas, W. Huhn, V. Blum, S. Goedecker, L. Frediani.
*The Elephant in the Room of Density Functional Theory Calculations.* J. Phys. Chem. Lett.
**8**(7), 1449–1457 (2017). DOI: 10.1021/acs.jpclett.7b00255. Code papers: P. Wind et al., *MRChem
Multiresolution Analysis Code for Molecular Electronic Structure Calculations: Performance and
Scaling Properties*, J. Chem. Theory Comput. **19**, 137 (2023), DOI: 10.1021/acs.jctc.2c00982; R.
J. Harrison et al., *MADNESS: A Multiresolution, Adaptive Numerical Environment for Scientific
Simulation*, SIAM J. Sci. Comput. **38**, S123 (2016), DOI: 10.1137/15M1026171. Multiwavelet (MRA)
all-electron DFT with guaranteed μHa precision; PBE and PBE0 references on 211 molecules, against
which triple-ζ Gaussian sets carry total-energy errors of about 10 kcal/mol.
**[USE]** Both a **novelty risk** for the "basis-set-free densities which no code can supply"
claim (kill-check 5, `01_PROJECT.md` Part F §5) and the best available **oracle for Increment 3**
— He and H₂ at LDA/PBE, all-electron, no basis. Not differentiable, not torch: an oracle, never a
substrate.

**[D23]** S. F. Boys, N. C. Handy. *The determination of energies and wavefunctions with full
electronic correlation.* Proc. R. Soc. Lond. A **310**(1500), 43–61 (1969). DOI:
10.1098/rspa.1969.0061
**[USE]** The transcorrelated similarity transform `f⁻¹Hf`, `f = e^{−u}`; the cusp factorisation
of D-35 is its one-body Jastrow special case, `u = Σ_a Z_a s_a`. Search for prior one-body
similarity transforms in grid DFT before claiming the technique.

---

## E. Pseudopotentials

**[E1]** D. R. Hamann. *Optimized norm-conserving Vanderbilt pseudopotentials.* Phys. Rev. B
**88**, 085117 (2013); Erratum Phys. Rev. B **95**, 239906 (2017), DOI:
10.1103/PhysRevB.95.239906. arXiv:1306.4707. DOI: 10.1103/PhysRevB.88.085117
**[USE — THE PSEUDOPOTENTIAL FORM IMPLEMENTED HERE]** ONCV: multiple projectors per angular
momentum, norm-conserving, separable Kleinman–Bylander form, soft enough for a real-space grid;
the v1 floor. The erratum matters — it corrects the published formulae.

**[E2]** M. Schlipf, F. Gygi. *Optimization algorithm for the generation of ONCV
pseudopotentials.* Comput. Phys. Commun. **196**, 36 (2015). DOI: 10.1016/j.cpc.2015.05.011
**[USE]** The SG15 table itself — the pseudopotential files consumed here (UPF format, LDA and PBE
variants, scalar- and fully-relativistic).

**[E3]** M. J. van Setten, M. Giantomassi, E. Bousquet, et al. *The PseudoDojo: Training and
grading a 85 element optimized norm-conserving pseudopotential table.* Comput. Phys. Commun.
**226**, 39 (2018). arXiv:1710.10138. DOI: 10.1016/j.cpc.2018.01.012
**[USE — VALIDATION DATA SOURCE]** A *graded* second ONCV table: each pseudopotential ships with
ghost-state tests, Δ-gauge scores and recommended cutoffs. Gate **G4.1** compares the solver's
atomic eigenvalues against the reference values in these files, proving the projector
implementation before any molecule.

**[E4]** L. Kleinman, D. M. Bylander. *Efficacious Form for Model Pseudopotentials.* Phys. Rev.
Lett. **48**, 1425 (1982). DOI: 10.1103/PhysRevLett.48.1425
**[USE]** The separable form that makes the non-local potential an O(N·N_proj) operation instead
of a dense operator; the ghost-state pathology it can introduce is checked by gates **G0.7** and
**G4.1**.

**[E5]** N. Troullier, J. L. Martins. *Efficient pseudopotentials for plane-wave calculations.*
Phys. Rev. B **43**, 1993 (1991). DOI: 10.1103/PhysRevB.43.1993
**[USE]** The classic smooth norm-conserving construction, the generator used in [D1]; fallback if
an element is missing from SG15/PseudoDojo.

**[E6]** P. E. Blöchl. *Projector augmented-wave method.* Phys. Rev. B **50**, 17953 (1994). DOI:
10.1103/PhysRevB.50.17953
*Deferred* (excluded from the v1 floor). Recorded because the GPAW fallback oracle of gates
**G4.3**/**G4.4** when Octopus is unavailable (risk R6 of `04_ROADMAP.md`; D-09) is PAW-based, and
the difference must be understood when interpreting disagreement.

**[E7]** D. Vanderbilt. *Soft self-consistent pseudopotentials in a generalized eigenvalue
formalism.* Phys. Rev. B **41**, 7892 (1990). DOI: 10.1103/PhysRevB.41.7892
*Deferred.* Ultrasoft potentials would cut grid cost substantially but introduce a generalised
eigenproblem and augmentation charges; the natural performance upgrade after v1.

---

## F. Numerical methods: eigensolvers, SCF acceleration, electrostatics

**[F1]** Y. Zhou, Y. Saad, M. L. Tiago, J. R. Chelikowsky. *Parallel self-consistent-field
calculations via Chebyshev-filtered subspace acceleration.* Phys. Rev. E **74**, 066704 (2006).
DOI: 10.1103/PhysRevE.74.066704
**[USE — PRIMARY EIGENSOLVER]** Chebyshev-filtered subspace iteration: needs only the action H·Ψ,
does almost all its work in dense GEMM on a block of vectors, needs no preconditioner, and leaves
only an N_states × N_states Rayleigh–Ritz step to be done in double precision.

**[F2]** Y. Zhou, Y. Saad, M. L. Tiago, J. R. Chelikowsky. *Self-consistent-field calculations
using Chebyshev-filtered subspace iteration.* J. Comput. Phys. **219**, 172 (2006). DOI:
10.1016/j.jcp.2006.03.017
**[USE]** The algorithmic companion to [F1]: filter degree selection, spectral bounds from a few
Lanczos steps, and the "filter once per SCF step" strategy that collapses the inner/outer loop.

**[F3]** Y. Zhou, J. R. Chelikowsky, Y. Saad. *Chebyshev-filtered subspace iteration method free
of sparse diagonalization for solving the Kohn–Sham equation.* J. Comput. Phys. **274**, 770
(2014). DOI: 10.1016/j.jcp.2014.06.056
**[USE]** Removes the remaining explicit diagonalisation; relevant at the 100-atom end of the
range.

**[F4]** A. V. Knyazev. *Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block
Preconditioned Conjugate Gradient Method.* SIAM J. Sci. Comput. **23**, 517 (2001). DOI:
10.1137/S1064827500366124
**[USE]** LOBPCG, the secondary eigensolver, kept as an independent implementation so gate
**G2.5** can be checked with two algorithms: disagreement localises the bug to the solver rather
than the Hamiltonian.

**[F5]** E. Vecharynski, C. Yang, J. E. Pask. *A projected preconditioned conjugate gradient
algorithm for computing many extreme eigenpairs of a Hermitian matrix.* J. Comput. Phys. **290**,
73 (2015). DOI: 10.1016/j.jcp.2015.02.030
**[USE]** PPCG; better than LOBPCG when many states are needed, a candidate upgrade at 100 atoms.

**[F6]** E. R. Davidson. *The iterative calculation of a few of the lowest eigenvalues and
corresponding eigenvectors of large real-symmetric matrices.* J. Comput. Phys. **17**, 87 (1975).
DOI: 10.1016/0021-9991(75)90065-0
**[USE]** Reference implementation for small test systems, where a textbook-simple solver makes
debugging tractable.

**[F7]** M. P. Teter, M. C. Payne, D. C. Allan. *Solution of Schrödinger's equation for large
systems.* Phys. Rev. B **40**, 12255 (1989). DOI: 10.1103/PhysRevB.40.12255
**[USE]** The TPA kinetic-energy preconditioner used by [F4]/[F5].

**[F8]** P. Pulay. *Convergence acceleration of iterative sequences. The case of SCF iteration.*
Chem. Phys. Lett. **73**, 393 (1980). DOI: 10.1016/0009-2614(80)80396-4
**[USE — PRIMARY MIXER]** DIIS/Pulay mixing on the density residual.

**[F9]** A. S. Banerjee, P. Suryanarayana, J. E. Pask. *Periodic Pulay method for robust and
efficient convergence acceleration of self-consistent field iterations.* Chem. Phys. Lett.
**647**, 31 (2016). DOI: 10.1016/j.cplett.2016.01.033
**[USE]** Periodic Pulay — Pulay extrapolation only every k-th step: cheap, markedly more robust,
the default in SPARC [D9], and the default mixer configuration here.

**[F10]** D. G. Anderson. *Iterative Procedures for Nonlinear Integral Equations.* J. ACM **12**,
547 (1965). DOI: 10.1145/321296.321305
**[USE]** Anderson mixing — the same family as [F8], with a cleaner formulation to implement and a
literature on convergence guarantees.

**[F11]** G. P. Kerker. *Efficient iteration scheme for self-consistent pseudopotential
calculations.* Phys. Rev. B **23**, 3082 (1981). DOI: 10.1103/PhysRevB.23.3082
**[USE]** Kerker preconditioning of the density residual: largely irrelevant for isolated
molecules (charge sloshing is a long-wavelength, metallic pathology), so implemented and *off by
default*, with the rationale recorded so that it is not switched on by cargo cult.

**[F12]** S. Lehtola, F. Blockhuys, C. Van Alsenoy. *An Overview of Self-Consistent Field
Calculations Within Finite Basis Sets.* Molecules **25**, 1218 (2020). DOI:
10.3390/molecules25051218
**[USE]** Systematic review of SCF convergence strategies and failure modes; the source of the
fallback ladder in `scf/loop.py` (DIIS → damped DIIS → linear mixing → level shift).

**[F13]** S. Lehtola. *Assessment of Initial Guesses for Self-Consistent Field Calculations.
Superposition of Atomic Potentials: Simple yet Efficient.* J. Chem. Theory Comput. **15**, 1593
(2019). DOI: 10.1021/acs.jctc.8b01089
**[USE]** The SAP initial guess: the cheapest reliable starting density, and the baseline that the
SCF-acceleration variant of the downstream target ("do better than SAP") must beat.

**[F14]** G. J. Martyna, M. E. Tuckerman. *A reciprocal space based method for treating long range
interactions in ab initio and force-field-based calculations in clusters.* J. Chem. Phys. **110**,
2810 (1999). DOI: 10.1063/1.477923
**[USE]** Removes periodic images from an FFT-based Hartree solve for a cluster; the pragmatic
alternative to [D14] when the implementation stays entirely inside `torch.fft`.

**[F15]** C. A. Rozzi, D. Varsano, A. Marini, E. K. U. Gross, A. Rubio. *Exact Coulomb cutoff
technique for supercell calculations.* Phys. Rev. B **73**, 205119 (2006). DOI:
10.1103/PhysRevB.73.205119
**[USE]** Spherical Coulomb cutoff kernel — the simplest correct open-boundary Hartree solver in
reciprocal space, implemented first because it is ten lines of torch and exactly verifiable
against a Gaussian charge distribution (gate **G0.2**).

**[F16]** P. García-Risueño, J. Alberdi-Rodriguez, M. J. T. Oliveira, X. Andrade, M. Pippig, J.
Muguerza, A. Arruabarrena, A. Rubio. *A survey of the parallel performance and accuracy of Poisson
solvers for electronic structure calculations.* J. Comput. Chem. **35**(6), 427–444 (2014).
arXiv:1211.2092 (2012). DOI: 10.1002/jcc.23487
**[USE]** Head-to-head accuracy comparison of [D14], [F14] and [F15]; used to justify the choice
and set the gate threshold.

**[F17]** N. D. Mermin. *Thermal Properties of the Inhomogeneous Electron Gas.* Phys. Rev.
**137**, A1441 (1965). DOI: 10.1103/PhysRev.137.A1441
**[USE]** Finite-temperature DFT: deferred, but the formal justification for the fractional
occupations carried from day one for the fractional-electron scenarios of [C1].

**[F18]** M. Methfessel, A. T. Paxton. *High-precision sampling for Brillouin-zone integration in
metals.* Phys. Rev. B **40**, 3616 (1989). DOI: 10.1103/PhysRevB.40.3616
**[USE]** Methfessel–Paxton smearing. Implemented, off by default.

**[F19]** N. Marzari, D. Vanderbilt, A. De Vita, M. C. Payne. *Thermal Contraction and Disordering
of the Al(110) Surface.* Phys. Rev. Lett. **82**, 3296 (1999). DOI: 10.1103/PhysRevLett.82.3296
**[USE]** Cold smearing; the variational-free-energy formulation matters if smearing is ever
enabled alongside forces.

**[F20]** F. J. dos Santos, N. Marzari. *Fermi energy determination for advanced smearing
techniques.* Phys. Rev. B **107**, 195122 (2023). arXiv:2212.07988. DOI:
10.1103/PhysRevB.107.195122
**[USE]** Robust bisection for the Fermi level with non-monotonic smearing functions — a real
source of silent failure once occupations are fractional.

**[F21]** J. F. Janak. *Proof that ∂E/∂n_i = ε_i in density-functional theory.* Phys. Rev. B
**18**, 7165 (1978). DOI: 10.1103/PhysRevB.18.7165
**[USE]** Janak's theorem becomes gate **G2.2**: a finite-difference derivative of the total
energy with respect to an occupation number must return that orbital's eigenvalue — a cheap
end-to-end check on the energy expression, the occupations and the eigensolver at once.

**[F22]** J. Harris. *Simplified method for calculating the energy of weakly interacting
fragments.* Phys. Rev. B **31**, 1770 (1985), DOI: 10.1103/PhysRevB.31.1770; W. M. C. Foulkes, R.
Haydock, *Tight-binding models and density-functional theory*, Phys. Rev. B **39**, 12520 (1989),
DOI: 10.1103/PhysRevB.39.12520.
**[USE]** The Harris–Foulkes functional; its difference from the self-consistent total energy is a
sharp, well-understood convergence diagnostic → gate **G2.1**.

**[F23]** C. Barat, A. Levitt, M. Torrent. *Preconditioning Magnetic Systems in Kohn-Sham Density
Functional Theory.* arXiv:2606.26693 (2026).
**[USE]** Relevant when spin polarisation is enabled (Increment 4); spin-channel preconditioning
is a known convergence trap.

**[F24]** T. Helgaker, P. Jørgensen, J. Olsen. *Molecular Electronic-Structure Theory.* Wiley
(2000), ch. 8 (Gaussian basis sets) — from standard knowledge.
**[USE]** Why the PySCF cc-pVQZ/5Z/aug-5Z totals of `cdft.reference.computed` carry a basis-set
truncation error of order 1e-4 Ha for He and H₂, and why a grid code compares *within* that
uncertainty and never tighter (G4.7, O-12 → O-20).

**[F25]** A. D. Becke. *A multicenter numerical integration scheme for polyatomic molecules.* J.
Chem. Phys. **88**(4), 2547–2553 (1988). DOI: 10.1063/1.454033
**[USE]** Fuzzy Voronoi cells (the iterated `p(μ) = 3μ/2 − μ³/2` step) and the radial map `r = r_m
(1 + x)/(1 − x)`: the atom-centred product grids on which `figures.py` integrates, independently
of the solver's own cusp-aware quadrature (D-54). Unadjusted cells only — a heteronuclear record
would need Becke's atomic-size adjustment, which is not implemented; the two quadratures agree on
`E_H + E_xc` of H₂⁺ to 7e-10 Ha (INC-F1), a cross-check of both.

---

## G. Kohn–Sham inversion (required for potential-target learning)

**[G1]** Q. Wu, W. Yang. *A direct optimization method for calculating density functionals and
exchange-correlation potentials from electron densities.* J. Chem. Phys. **118**, 2498 (2003).
DOI: 10.1063/1.1535422
**[USE]** The standard direct-optimisation inversion: given a reference density it recovers v_s(r)
and hence v_xc(r) — the targets that [C13]-style potential learning needs.

**[G2]** Q. Zhao, R. C. Morrison, R. G. Parr. *From electron densities to Kohn-Sham kinetic
energies, orbital energies, exchange-correlation potentials, and exchange-correlation energies.*
Phys. Rev. A **50**, 2138 (1994). DOI: 10.1103/PhysRevA.50.2138
**[USE]** The ZMP constrained-search method; simpler to implement and a good cross-check on [G1].

**[G3]** Y. Shi, V. H. Chávez, A. Wasserman. *n2v: A density-to-potential inversion suite. A
sandbox for creating, testing, and benchmarking density functional theory inversion methods.*
WIREs Comput. Mol. Sci. **12**, e1617 (2022). DOI: 10.1002/wcms.1617
**[USE]** Reference implementation and a catalogue of the numerical pathologies inversion is prone
to (basis-set artefacts, oscillations near nuclei).

**[G4]** M. F. Herbst, V. H. Bakkestuen, A. Laestadius. *Kohn–Sham inversion with mathematical
guarantees.* Phys. Rev. B **111**, 205143 (2025). arXiv:2409.04372. DOI:
10.1103/PhysRevB.111.205143 (demonstrated for periodic bulk Si, GaAs and KCl)
**[USE]** Moreau–Yosida regularised inversion with convergence guarantees — the mathematically
clean route, and the one most compatible with a WPI/Mauser-group thesis.

**[G5]** V. H. Bakkestuen, M. F. Herbst, V. Falmår, M. Penz, A. Laestadius. *Moreau–Yosida-based
Kohn–Sham Inversion for Periodic Systems.* arXiv:2606.19471 (2026).
**[USE]** Extension of [G4].

**[G6]** Z. Chai, S. Luber. *Stable, Fast, and Accurate Kohn-Sham Inversion in Gaussian Basis for
Open Shell Molecular and Condensed Phase Systems via Density Matrix Penalization.*
arXiv:2603.22140 (2026).
**[USE]** Current state of the art for inversion stability; read before implementing Increment 10.

**[G7]** O. M. Bohle, M. Lotfigolian, A. Laestadius, E. I. Tellgren. *Regularised
density-potential inversion for periodic systems: application to exact exchange in one dimension.*
arXiv:2510.24330 (2025).
**[USE]** Regularisation strategy plus a 1D test case with an exact answer — the inversion unit
test.

**[G8]** S. Nam, S. Song, E. Sim, K. Burke. *Measuring Density-Driven Errors Using Kohn–Sham
Inversion.* J. Chem. Theory Comput. **16**(8), 5014–5023 (2020). arXiv:2004.11595. DOI:
10.1021/acs.jctc.0c00391
**[USE]** The operational bridge between [B7] and [G1]: how to use inversion to *measure* the
density-driven component, the quantity a corpus must make computable if a model is to learn the
functional error alone.

**[G9]** V. Subramanian, B. Kanungo, V. Gavini. *invDFT: A CPU-GPU massively parallel tool to find
exact exchange-correlation potentials from groundstate densities.* Comput. Phys. Commun. **326**,
110218 (2026). arXiv:2510.10529. DOI: 10.1016/j.cpc.2026.110218
**[USE]** The real-space (finite-element) inversion code of the novelty audit (`01_PROJECT.md`
Part F §2): open source (LGPL-2.1), reads densities from Q-Chem, NWChem and PySCF, demonstrated on
FCI and model densities of systems with up to 100 electrons, spin-restricted so far. Read before
Increment 10.

**[G10]** B. Kanungo, P. M. Zimmerman, V. Gavini. *Exact exchange-correlation potentials from
ground-state electron densities.* Nat. Commun. **10**, 4497 (2019), DOI:
10.1038/s41467-019-12467-0; and *A Comparison of Exact and Model Exchange–Correlation Potentials
for Molecules*, J. Phys. Chem. Lett. **12**(50), 12012–12019 (2021), DOI:
10.1021/acs.jpclett.1c03670. Follow-up: V. Subramanian et al., *Exchange-Correlation Potentials
and Energies from Inverse Generalized Kohn-Sham Calculations*, arXiv:2609.15845 (2026).
**[USE]** The inversion method behind [C19] and [G9]; the 2026 follow-up shows KS and
generalized-KS correlation quantities differ substantially in strongly correlated systems —
relevant to which target an inverted potential represents.

---

## H. Time propagation (architectural reserve, not v1)

**[H1]** E. Runge, E. K. U. Gross. *Density-Functional Theory for Time-Dependent Systems.* Phys.
Rev. Lett. **52**, 997 (1984). DOI: 10.1103/PhysRevLett.52.997
**[USE]** The theorem that makes the later TDDFT increment legitimate.

**[H2]** A. Castro, M. A. L. Marques, A. Rubio. *Propagators for the time-dependent Kohn–Sham
equations.* J. Chem. Phys. **121**, 3425 (2004). DOI: 10.1063/1.1774980
**[USE]** Every propagator needs only H·Ψ, so if `Hamiltonian.apply` is the only seam the SCF
uses, TDDFT is additive rather than a rewrite — the technical basis for "architect for TDDFT,
build ground state".

**[H3]** A. Gómez Pueyo, M. A. L. Marques, A. Rubio, A. Castro. *Propagators for the
Time-Dependent Kohn–Sham Equations: Multistep, Runge–Kutta, Exponential Runge–Kutta, and
Commutator Free Magnus Methods.* J. Chem. Theory Comput. **14**, 3040 (2018). DOI:
10.1021/acs.jctc.8b00197. arXiv:1803.02113
**[USE]** Modern propagator comparison, including commutator-free Magnus schemes — directly
relevant to the Mauser group's numerical-analysis interests.

---

## I. Software infrastructure and dependencies

**[I1]** S. Lehtola, C. Steigemann, M. J. T. Oliveira, M. A. L. Marques. *Recent developments in
libxc — A comprehensive library of functionals for density functional theory.* SoftwareX **7**, 1
(2018). DOI: 10.1016/j.softx.2017.11.002
**[USE — VERIFICATION ORACLE]** ~600 functionals with analytic derivatives, used as a *pointwise*
oracle (gate **G0.6**) against the torch implementations here and as a breadth fallback. Never on
the hot path: it is C, CPU-only and opaque to autograd, which [C3] shows is disqualifying
downstream.

**[I2]** Q. Sun, X. Zhang, S. Banerjee, et al. *Recent developments in the PySCF program package.*
J. Chem. Phys. **153**, 024109 (2020). DOI: 10.1063/5.0006074
**[USE]** Independent-implementation oracle for molecular reference numbers, and the code in which
DM21 [C1] and Skala [C2] are evaluated — the bridge if this project's densities are ever scored
with a published learned functional.

**[I3]** R. Li, Q. Sun, X. Zhang, G. K.-L. Chan. *Introducing GPU Acceleration into the
Python-Based Simulations of Chemistry Framework.* J. Phys. Chem. A **129**(5), 1459–1468 (2025).
arXiv:2407.09700. DOI: 10.1021/acs.jpca.4c05876
**[USE]** How a Python quantum-chemistry code actually gets GPU speedups, and where the Python
overhead bites. Read before Increment 6.

**[I4]** A. Hjorth Larsen, J. J. Mortensen, J. Blomqvist, et al. *The atomic simulation
environment — a Python library for working with atoms.* J. Phys.: Condens. Matter **29**, 273002
(2017). DOI: 10.1088/1361-648X/aa680e
**[USE]** Structure objects, file I/O and a calculator interface; implementing the ASE calculator
protocol makes geometry optimisation, MD and the GPAW/Octopus cross-checks free.

**[I5]** A. Paszke, S. Gross, F. Massa, et al. *PyTorch: An Imperative Style, High-Performance
Deep Learning Library.* NeurIPS (2019). arXiv:1912.01703
**[USE]** The compute substrate: autograd (required by [C3]/[C4]), `torch.fft`, batched linear
algebra, and one code path for CPU float64 and GPU float32.

**[I6]** P. Steinbach, C. Bannwarth. *Acceleration of Semiempirical Electronic Structure Theory
Calculations on Consumer-Grade GPUs Using Mixed-Precision Density Matrix Purification.* J. Chem.
Theory Comput. **21**(15), 7335–7351 (2025). DOI: 10.1021/acs.jctc.5c00262
**[USE — DIRECTLY ON POINT]** Mixed precision on *consumer-grade* GPUs — exactly the GTX 1660 Ti
situation; empirical support for the float32-hot-path / float64-reduction policy.

**[I7]** J. Woo, S. Kim, W. Y. Kim. *Dynamic Precision Approach for Accelerating Large-Scale
Eigenvalue Solvers in Electronic Structure Calculations on Graphics Processing Units.* J. Chem.
Theory Comput. **19**(5), 1457–1465 (2023). DOI: 10.1021/acs.jctc.2c00983. PMID: 36812094
**[USE]** Eigensolver iterations can start in low precision and tighten as they converge — a
concrete scheduling policy for the Chebyshev filter of [F1].

**[I8]** A. Alvermann, A. Basermann, H.-J. Bungartz, C. Carbogno, D. Ernst, H. Fehske, et al.
*Benefits from using mixed precision computations in the ELPA-AEO and ESSEX-II eigensolver
projects.* Japan J. Ind. Appl. Math. **36**(2), 699–717 (2019). arXiv:1806.01036. DOI:
10.1007/s13160-019-00360-8
**[USE]** Where mixed precision breaks eigensolvers — loss of orthogonality in the block — and
therefore why the Gram matrix and Rayleigh–Ritz step stay in float64 unconditionally.

**[I9]** HDF5 / h5py.
**[USE]** The corpus format: chunked, compressed, partially readable, language-agnostic, and able
to carry the provenance attributes that make a record auditable.

**[I10]** X. Wu, Q. Sun, Z. Pu, T. Zheng, W. Ma, W. Yan, et al. (14 authors). *Enhancing
GPU-Acceleration in the Python-Based Simulations of Chemistry Frameworks.* WIREs Comput. Mol. Sci.
**15**(2) (2025). arXiv:2404.09452. DOI: 10.1002/wcms.70008
**[USE]** GPU4PySCF v1.0 (GPU DFT with density fitting, ~30× over a 32-core CPU node); same
reading as [I3]: where the Python overhead bites on a GPU.

---

## J. Deliberate exclusions, and what is not covered

Four topics are absent on purpose: **periodic-solid literature** (v1 is 3D molecules, finite and
non-periodic — no Brillouin-zone sampling, symmetry reduction or band structures);
**linear-scaling / O(N) methods** (at 50–100 atoms the bottleneck is memory, not cubic scaling);
**exact exchange on a grid** (deferred with rung 4; the entry points when it is needed are the ACE
operator, Lin Lin, JCTC 2016, and the Octopus implementation [D5]); and **quantum-chemistry
integral literature** (McMurchie–Davidson, Obara–Saika — the Gaussian-basis route was considered
and rejected).

**[J1]** Physics-informed neural networks for quantum eigenvalue problems — e.g. H. Jin, M.
Mattheakis, P. Protopapas, *Physics-Informed Neural Networks for Quantum Eigenvalue Problems*,
IJCNN 2022, DOI: 10.1109/IJCNN55064.2022.9891944, arXiv:2203.00451; K. Shah, P. Stiller, N.
Hoffmann, A. Cangi, *Physics-Informed Neural Networks as Solvers for the Time-Dependent
Schrödinger Equation*, arXiv:2210.12522; S. Sarkar, *Physics-Informed Neural Networks for
One-Dimensional Quantum Well Problems*, arXiv:2504.05367. *Excluded from the baseline by mandate*:
the baseline is a state-of-the-art method from the literature, excluding PINNs. Listed so the
downstream project has its entry points and the boundary between this repository and the PINN
repository is written down rather than assumed.

**[J2]** Classical density functional theory of fluids (Evans-type cDFT/DDFT). *Not this project.*
"Classical" in the project name means *non-neural*; the source paper [A17] is unambiguously
electronic-structure DFT. Recorded because the folder name will mislead a future reader.

**[J3]** M. Z. Naser. *Fundamental flaws of physics-informed neural networks and explainability
methods in engineering systems.* Comput. Ind. Eng. **212**, 111704 (2026; online November 2025).
DOI: 10.1016/j.cie.2025.111704

**[J4]** C. Xu, D. Liu, A. Nassereldine, J. Xiong. *FP64 is All You Need: Rethinking Failure Modes
in Physics-Informed Neural Networks.* arXiv:2505.10949 (2025).

**[J5]** A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, M. W. Mahoney. *Characterizing
possible failure modes in physics-informed neural networks.* Advances in Neural Information
Processing Systems **34**, 26548–26560 (NeurIPS 2021). arXiv:2109.01050.
**[USE for J3–J5]** The critique literature on strict PINNs: not a reason to avoid
physics-constrained ML — [C3] is the version that works — but a reason to avoid the *label*; see
`01_PROJECT.md` Part D §2 and section L below.

---

## K. Industrial and applied context

Supporting `01_PROJECT.md` Part D §4. Cited for *where the modelling is used and which errors cost
money*, not for method. The sector map behind that section, kept here as its single home:

| Sector | What DFT computes | Which functional failure bites | Tags |
|---|---|---|---|
| batteries, energy storage | redox potentials, Li/Na barriers, electrolyte decomposition, SEI, polaron localisation | delocalization error in transition-metal oxides; SIE in charge-localised states | [K3], [K4] |
| heterogeneous catalysis | adsorption energies, surface barriers, site preference | barriers underestimated from SIE; the CO/Pt(111) site puzzle, resolved by [C20] | [K2], [K5] |
| semiconductors | band gaps, defect charge-transition levels, dopant activation, 2-D channels | gaps ~50 % low semi-locally; defect level placement | [K6], [K7] |
| pharmaceuticals | conformer and tautomer energies, protonation states, QM/MM barriers | non-covalent interactions, dispersion, barrier heights | [K8] |
| quantum technology | colour-centre zero-phonon lines, spin splittings, hyperfine couplings | SIE in strongly localised defect states; spin-state ordering | [K7] |
| photovoltaics, OLED | excitation energies, charge-transfer states, singlet–triplet gaps | charge-transfer excitations badly wrong semi-locally — a direct SIE consequence | — |
| magnets, spintronics | magnetic moments, exchange couplings, spin-state energetics | transition-metal spin-state ordering | — |
| carbon capture | CO₂ binding in MOFs and amine sorbents | dispersion plus open-metal-site transition metals | [K9] |
| nuclear, fusion materials | defect energetics under irradiation, He bubbles, molten-salt chemistry | magnetic Fe and W; strongly correlated defects | — |

On the critical path of this modelling: solid-state and sodium-ion batteries; iridium-free
oxygen-evolution catalysts; 2-D-material transistors [K6]; ferroelectric and memristive memory;
colour-centre qubits and sensors [K7]; direct air capture; sustainable fuels and plastic upcycling;
fusion first-wall materials; and quantum computing, where classical DFT plus wavefunction baselines
are the yardstick for any claimed advantage. Machine-learned interatomic potentials have absorbed
most throughput use cases [C25] but inherit whatever functional generated their data.

**[K1]** P. B. Shukla, P. Mishra, T. Baruah, R. R. Zope, K. A. Jackson, J. K. Johnson. *How Do
Self-Interaction Errors Associated with Stretched Bonds Affect Barrier Height Predictions?* J.
Phys. Chem. A **127**(7), 1750–1759 (2023). DOI: 10.1021/acs.jpca.2c07894. PMID: 36787213.
**[USE]** Traces barrier-height error quantitatively to self-interaction at stretched bonds — the
mechanism behind the Arrhenius rate table of `01_PROJECT.md` Part D.

**[K2]** P. B. Shukla, S. Romero, T. Baruah, R. R. Zope, K. A. Jackson, J. K. Johnson. *Assessing
Self-Interaction Corrections in the Selective Catalytic Reduction of NO on a Cu-SSZ-13 Zeolite
Cluster Model.* J. Phys. Chem. A **130**(30), 5840–5851 (2026). DOI: 10.1021/acs.jpca.6c02902.
**[USE]** The same error on a real industrial catalyst (diesel emissions aftertreatment): evidence
that SIE is a commercial problem, not an academic one.

**[K3]** Microsoft Research. *MatterGen* — C. Zeni, R. Pinsler, D. Zügner, et al. (26 authors), *A
generative model for inorganic materials design*, Nature **639**, 624–632 (2025), DOI:
10.1038/s41586-025-08628-5 — and *MatterSim* — H. Yang, C. Hu, Y. Zhou, et al., *MatterSim: A Deep
Learning Atomistic Model Across Elements, Temperatures and Pressures*, arXiv:2405.04967 (2024).
Project pages: microsoft.com/en-us/research/project/materials/; github.com/microsoft/mattergen.
**[USE]** Generation is cheap, verification is the bottleneck: every generative materials pipeline
terminates in a physics calculation whose trustworthiness sets the pipeline's false-positive rate.

**[K4]** V. Ragavendran. *Machine learning-accelerated discovery of battery electrode materials
for sustainable energy storage: A critical survey of DFT descriptors, benchmarked models and
experimentally validated predictions in support of SDG 7 and SDG 9.* Computational Condensed
Matter **48**, e01410 (2026). DOI: 10.1016/j.cocom.2026.e01410. See also K. S. Kim, *Machine
Learning for Accelerating Energy Materials Discovery: Bridging Quantum Accuracy with Computational
Efficiency*, Adv. Energy Mater. **16**(2) (2026), DOI: 10.1002/aenm.202503356.
**[USE]** The battery-materials pipeline and which DFT quantities it actually consumes.

**[K5]** *Density Functional Theory for Molecule–Metal Surface Reactions: When Does the
Generalized Gradient Approximation Get It Right, and What to Do If It Does Not.* N. Gerrits, E. W.
F. Smeets, S. Vuckovic, A. D. Powell, K. Doblhoff-Dier, G.-J. Kroes. J. Phys. Chem. Lett.
**11**(24), 10552–10560 (2020). DOI: 10.1021/acs.jpclett.0c02452
**[USE]** The accuracy actually required for heterogeneous catalysis modelling, and where GGA
fails.

**[K6]** imec, ASML and TSMC: 300 mm integration route for industry-ready 2-D-material-based
transistors (2026); imec record WSe₂ 2-D pFETs (2025). imec-int.com press releases. *(Not
verifiable on 2026-09-16: the imec press pages could not be retrieved — titles and dates must be
confirmed before this entry is cited.)*
**[USE]** 2-D-material devices are on an industrial roadmap, which makes their interface and
defect physics a first-principles question with a delivery date.

**[K7]** S. Li, A. Gali, B. Huang. *Computation-aided design of color centers for quantum
information processing.* Newton **2**(5), 100432 (2026). DOI: 10.1016/j.newton.2026.100432. See
also J. R. Weber, W. F. Koehl, J. B. Varley, A. Janotti, B. B. Buckley, C. G. Van de Walle, D. D.
Awschalom, *Quantum computing with defects*, PNAS **107**, 8513 (2010), DOI:
10.1073/pnas.1003052107, and A. Bilgin, I. N. Hammock, J. Estes, Y. Jin, H. Bernien, A. A. High,
G. Galli, *Donor–acceptor pairs in wide-bandgap semiconductors for quantum technology
applications*, npj Comput. Mater. **10**, 7 (2024), DOI: 10.1038/s41524-023-01190-6.
**[USE]** Designing a defect with specified optical and spin properties is something only first
principles can do — and localised defect states are exactly where self-interaction error bites.

**[K8]** Schrödinger, Inc. (NASDAQ: SDGR), *Update on Progress Across the Business and 2026
Strategic Priorities* (January 2026).
**[USE]** A publicly traded company whose product is physics-based molecular simulation; the
existence of the business model is the industry-legitimacy datum.

**[K9]** T. Jaffrelot Inizan, P. D. Kamath, A. M. Elena, K. A. Persson. *uMOF: A Universal
Database, Benchmark, and Machine Learning Interatomic Potentials for Metal-Organic Frameworks.*
arXiv:2608.28100 (2026) (r2SCAN-D4 data for 19,950 MOFs).
**[USE]** Carbon capture and gas separation; the sorbent-design window is a binding-energy
problem.

---

## Verification status

**Confirmed against the primary source on 2026-09-16.** Every tagged entry was re-checked on
2026-09-15/16 — the last full pass — against live publisher pages where reachable, arXiv, NIST,
the DOI Foundation citation formatter, OpenAlex and Europe PMC/PubMed. Where a publisher refuses
automated access (APS, ACS, AIP, Wiley, Science, IOP) the metadata comes from the DOI, OpenAlex or
Europe PMC record; no field was filled from memory. Every correction from that pass is applied in
the entry itself, and the entry states what was wrong wherever the incorrect version may have been
quoted elsewhere ([A17], [C2], [C8]/[C8b], [C19], [C20], [C22], [C27], [D4], [D17], [D18], [G3],
[I3], [K7]). The eight figure-diagnostics entries added on 2026-09-17 ([A29]–[A34], [B11], [F25])
were each read off a live OpenAlex, Crossref or PubMed record that day.

**Not yet confirmed against the primary source.** These details are carried as transcribed and
must not be quoted with more confidence than this table states.

| Entry | Detail | Status |
|---|---|---|
| [A17] | the paraphrase about large, diverse datasets of near-exact reference results | not yet confirmed; kept, and flagged in the entry |
| [B5] | the numerical "excellent/good" cut-offs for ε and ν | not yet confirmed; the supplementary section stating them could not be retrieved, and the transcribed values should be checked against the published table before being relied on |
| [B6] | >100M or >140M single-point calculations | version-dependent wording; the cited version must be stated |
| [C24] | the ICML 2026 venue | from the PDF footer only; not independently confirmed |
| [D9] | the SPARC v2.0.0 program paper | OpenAlex record only |
| [G3] | the article number e1617 | not yet confirmed; volume, issue and DOI are confirmed |
| [K4] | the Adv. Energy Mater. article number | OpenAlex record only |
| [K6] | the imec press releases | not retrievable on 2026-09-16 |

**Cited at textbook confidence, not re-verified line by line.** [A1]–[A3], [A5]–[A7], [A16],
[A18]–[A20], [B9], [B10], [D8], [E4]–[E7], [F3], [F5]–[F8], [F10], [F11], [F17], [F22], [F24],
[G2], [H1], [I4], [I5], [I9]. These need a reference-manager check before they are carried into a
bibliography.

The knock-on corrections this pass implies for `01_PROJECT.md` (Part B §2), `03_METHOD.md` (G4.2)
and `05_DECISION_LOG.md` (D-01) were applied on 2026-09-17; anything quoting an entry listed above
as not yet confirmed is tracked in `02_STATUS.md` Part A §7.

---

## L. What the machine-learned-functional literature implies for this project

Sections A–K record what each source contributes to a subsystem. This section reads the
learned-exchange–correlation literature as a whole back onto *this* instrument — a real-space,
all-electron, gate-verified KS solver for H, He⁺, He, H₂⁺ and H₂ whose records are the baseline
and the labels for a learned functional or PINN built later (`01_PROJECT.md` Parts A, C, D). It
rests on a survey of that literature (roughly 200 sources, current to 2026-09-17) carried out for
this project, whose open questions are numbered Q1–Q23 and whose research directions are numbered
RD 1–RD 9; those numbers are used below. Quoted figures keep the hedges of the sources they came
from.

### 1. The four roles of ML in a Kohn–Sham calculation, and which one this project serves

| Role | What it does | This project |
|---|---|---|
| **A — replace the unknown** | a learned E_xc or v_xc inside the cycle (DM21 [C1], Skala [C2], CIDER [C26]/[C27]); *changes the answer* | out of scope by mandate (gate G5.7); the corpus is what such a model is trained on |
| **B — accelerate the known** | learned starting densities, density matrices, mixing ([C22], [C23]); *same answer, fewer iterations* | the safe win: it changes only the path to the fixed point, so gate G3.4 verifies it outright |
| **C — bypass the cycle** | predict Hamiltonians, densities or energies directly; *only as good as its training data* | not a route taken here; this instrument is the data such a model would inherit |
| **D — produce the labels** | coupled cluster, NN wavefunctions [C24], KS inversion (section G), exact conditions; *defines "accurate"* | **this project's role** — and a second one the taxonomy does not draw: a verification instrument |

That taxonomy settles the priority: only role A changes the self-consistent result, but role D
decides what accurate means for the other three. A gate-certified record with its density, its
potential and its gate report is a role-D artefact; a fast but unverified one is not.

### 2. Free exact labels at fractional occupation

Piecewise linearity [A21] and the flat plane [A23]/[A24] make the exact energy at fractional N and
M a straight line between integer endpoints — a free source of labels, *data augmentation from an
exact property of the target*. DM21 used it for 1 074 fractional-charge and fractional-spin
systems [C1], and a 1D ensemble model has been shown to learn the derivative discontinuity from
it. Flat-plane tests of the newest functionals have not been reported (open question Q5, research
direction 2), and the condition has hardly been examined away from training data as of September
2026. **For this project:** the fractional-N
scans of D-22 and diagnostics D1.2–D1.4 are exactly that missing instrument, and they need no
correlated reference — only the integer endpoints this solver already produces to 1e-13 Ha for H
and He⁺.

### 3. Potentials as targets

With one occupied spatial orbital the inversion is closed form, `v_s = ε + ∇²√n/(2√n)` — every
Phase 1 system is in that class, so I10 delivers exact `v_s` with no ill-posed optimisation.
Beyond one orbital, inversion is ill-posed and finite bases make it worse ([G3]; likewise for
orbital-averaged inversion at integer and fractional N), which is why
basis-set-free real-space densities are the natural input ([G9], spin-restricted). Kanungo 2025
[C19] trained neural LDA/GGA functionals on exact potentials of seven small systems (Li, C, N, O,
Ne; H₂, LiH) to SCAN-level accuracy: potential targets are the high-leverage route for a group
without industrial compute. **Consequence:** the corpus stores `v_s`, `v_xc`, `n`, `∇n` and `τ` on
the grid, not summaries (`03_METHOD.md` Part A §7), and I10 is core rather than optional (D-17).

### 4. Why basis-set-free real-space data matter

Most learned functionals were developed in Gaussian bases. Against multiwavelet references for 211
molecules, triple-ζ Gaussian sets carry total-energy errors of about 10 kcal/mol [D22]. It is open
(Q13) whether learned functionals depend on the basis or grid used in training and whether they
converge under refinement — no systematic study of this was found — and (Q15, research direction
9) cross-code verification exists for PBE on solids ([B4], [B5]) while no comparable study of a
learned functional was found, their implementations meanwhile multiplying ([C21], itself a
preprint). The numerical literature is explicit that grid artefacts must be controlled before
small learned corrections can be judged. **That is this project's contribution:** the
egg-box gate G2.7 [D17], the grid-convergence gate G3.1, the ε/ν cross-code construction of
G4.3/G4.4 [B5], and the numerical standard of `02_STATUS.md` Part A §2 are the argument, not
overhead.

### 5. Self-interaction is still the weak spot

SIE4x4 remains among the largest GMTKN55 subset errors of the best learned functional — 13.6
kcal/mol for Skala-1.0 [A17] — and the reason is understood: the Hartree energy is a non-local
functional, so a *semilocal* model cannot in general cancel the self-repulsion of an arbitrary
one-electron density, such as that of stretched H₂⁺. Open question Q6 asks for the minimal
non-locality that removes delocalisation error and at what cost; [C28] shows the flat plane is
recoverable at semi-local *cost*, and [C27] gives the Θ(N log N) FFT route for non-local density
features. The project's SIE programme (`01_PROJECT.md` Part C) therefore targets a live gap on the
substrate where those features are cheapest, with stretched H₂⁺ as the cheapest exact test of it.

### 6. PINN cautions to carry downstream

Carry four findings into the PINN repository, not into this one ([J3]–[J5]): double precision
removes many reported PINN failures [J4] — which makes this solver's float64 reference path an
asset rather than a luxury; curriculum training lowers errors by one to two orders of magnitude
[J5]; a published survey of the field found that 79 % (60 of 76) of papers claiming to beat
standard numerical methods used a weak baseline, so the baseline a learned model must beat is a
gate-certified record; and a strict orbital-PINN — a network representing many mutually orthogonal
orbitals trained on the eigenvalue residual — would compete with highly optimised eigensolvers
([F1], [F4]). The physics-informed ideas that actually worked in DFT are of a different kind: the
KS equations as a regulariser [C3], self-consistency as a label-free training signal, and
variational training without labels.

### 7. Energy-only training cannot separate the two errors

The functional-driven / density-driven split [B7] is open question Q12: the split is known and its
pitfalls are documented [B8], but it is rarely applied to learned functionals. A model trained on
energies at fixed densities learns the *sum*. Inversion measures the density-driven part [G8].
**Consequence for the schema:** densities and potentials are stored with every energy, which is
also what makes the corpus usable for research direction 1 (independent benchmarks of released
functionals with a density-driven-error analysis).

### 8. What this instrument can address

| Item | The question | Where it lands here |
|---|---|---|
| Q5 | is exact information at fractional occupation used to its potential? | D-22 scans; diagnostics D1.2–D1.4 |
| Q8 | steps, discontinuities, asymptotics of exact potentials | D1.6/D1.7 ([A30], [A31]); I10 |
| Q11 | potentials smooth enough for stable self-consistency | records keep the full SCF trajectory and `v_xc`; gates G2.x/G3.x |
| Q12 | the density-driven share of a learned functional's error | corpus stores `n` and the potentials; [B7], [G8] |
| Q13 | basis/grid dependence of learned functionals | G3.1, G2.7 and the numerical standard |
| Q15 | cross-code verification of learned functionals | G4.3/G4.4 against Octopus [D5] and GPAW [D7], ε/ν of [B5] |
| Q23 | robust inversion | I10, one-orbital closed form first; [G4], [G9] |
| RD 2 | flat-plane diagnostics for learned functionals | D1.4 over the E(N, M) scans |
| RD 3 | potential targets at scale | I10 plus the corpus schema |
| RD 4 | non-local features on real-space grids, and their grid dependence | [C26]/[C27] on this substrate; the I9 records |
| RD 9 | cross-code verification protocols | G4.3/G4.4, built on [B5] |

The instrument does not address the data-scale questions (Q1–Q4, Q16–Q19): five systems at
1e-13 Ha are a verification and diagnostic corpus, not a training set for transferability.