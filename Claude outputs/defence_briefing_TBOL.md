**For** the supervisor discussion of the v2 two-pager. **Built on** the literature map of 22 September 2026
(161 checked sources) and the project's own band calculation. Numbers in the text are the ones worth
remembering; §8 collects them. Where a number comes from our own single-particle bands, which contain no
interactions, trap or temperature, the text says so. Where a step is ours rather than a source's, it says
that too, because you will be asked.

## 1. The system, in plain words

Take a Bose–Einstein condensate of ⁸⁷Rb atoms. Each atom can be in one of two internal states, call them
|1⟩ and |2⟩. Shine two square optical lattices onto the cloud, made from laser light at two "tune-out"
wavelengths chosen so that each lattice is felt by only one of the two states. Both lattices have the same
spacing, a = 395 nm, and the same natural energy unit, the recoil energy E_R = h × 3.67 kHz. Rotate them
against each other by θ = 5.21°. Finally apply a microwave field at the 6.8 GHz hyperfine transition, which
converts |1⟩ into |2⟩ and back at a rate Ω_R, the Rabi coupling, tunable from zero up to about 1 E_R.

Why is this called a twisted bilayer? Think of the internal state as a layer index: atoms in |1⟩ live in
layer 1 (its lattice), atoms in |2⟩ in layer 2 (the rotated lattice), and the microwave lets an atom hop
between the layers. That is the structure of twisted bilayer graphene, two rotated lattices coupled by
interlayer tunnelling. The differences are worth listing now, because the whole argument turns on them: the
layers are internal states of the same atoms in the same plane; the tunnelling is a microwave, the same
everywhere in space, rather than an overlap of orbitals that varies with position; the interaction is a
short-range contact term rather than Coulomb repulsion; and the particles are bosons.

The geometry you should be able to draw on a whiteboard: two square lattices rotated by θ are in register
in some places, with sites on top of each other (call these AA regions), and displaced by half a site in
others (AB regions). The pattern repeats with the moiré period L_M = a/(2 sin θ/2), which at 5.21° is
eleven lattice constants, 4.35 µm. Theory works at the nearby commensurate angle 2 arctan(1/22) = 5.205°,
where the two lattices share an exact supercell of 22 lattice constants and Bloch's theorem applies. Our own
calculations use the same family of angles, tan(θ/2) = 1/m, mostly at m = 8 (θ = 14.25°), because the
supercell is then small and a band structure takes seconds; the experiment's m = 22 has not been computed
yet.

The cloud is quasi-two-dimensional, a single pancake cut out by a tight vertical lattice (ω_z ≈ 2π × 1 kHz),
holding up to 5 × 10⁵ atoms. The paper treats the interactions as the same whichever states collide.

## 2. Bands: what happens to the single-particle spectrum

Start with one lattice. An atom in a periodic potential has Bloch states labelled by a crystal momentum k
and a band index. The lowest band, the s-band, is built from the ground state of each well; its width is 8J,
where J is the tunnelling between neighbouring sites. J falls exponentially with lattice depth: from the
exact band structure, J = 0.086, 0.031 and 0.012 E_R at V₀ = 4, 8 and 12 E_R. The paper uses the familiar
deep-lattice formula instead, which gives a hopping 36 % larger at 4 E_R; this matters because every
threshold below is quoted as a ratio Ω/J, and ours use the exact J. The next band lies about 3.7 E_R higher
at V₀ = 8. Bosons condense at the bottom of the lowest band, k = 0.

Now two rotated lattices, one per spin state, and a microwave Ω coupling the states. On the moiré supercell
the s-band of each layer is folded into many thin sub-bands, and at Ω = 0 the lowest of them is simply a
slice of the original s-band; nothing physical has happened. Switch on Ω. It converts an atom in a well of
lattice 1 into an atom at the same place in lattice 2, so the matrix element is the overlap of the two wells'
wavefunctions there: large where the wells coincide (AA), small where they are displaced (AB). For the
symmetric combination of the two states this acts as an attractive potential with the moiré period, deepest
at AA, whose depth grows linearly with Ω. This is the effective theory of Wang, Gao and Shi (2025), a
preprint written for a three-dimensional quasi-moiré crystal; applying it to the two-dimensional square
bilayer is our step, not theirs. Kariyado and Vishwanath (2019) reached the same conclusion from symmetry:
flat bands in twisted Bravais bilayers need a coupling that varies in space so that symmetry can force exact
cancellations, and a uniform coupling cannot do this. The Nature paper's own strong-coupling limit says the
same in its own words: the lower dressed state feels the average of the two lattices, minus Ω_R, and the
average is deepest at AA.

What our calculation shows, at m = 8 and V₀ = 8 E_R: the lowest band is untouched up to Ω/J ≈ 3, has lost
half its width by Ω/J ≈ 12, splits off from the band above it between Ω/J ≈ 10 and 16, and has collapsed by
more than two orders of magnitude by Ω/J ≈ 32. As it collapses, the atoms fall into the AA sites: the fraction
of the moiré cell the band-bottom state occupies drops from about 40 % to under 1 %, and the density ratio
between AA and AB regions rises past a hundred at Ω/J ≈ 16 and past a thousand at 32. At 12 E_R the same
happens at smaller Ω/J (halving at about 9), at 4 E_R only at much larger (about 35). In one sentence: the
"flat band" is an array of isolated pockets, one per moiré cell, produced by localisation, not by interference
at a magic angle. The phrase in the Nature paper, "a lowest flat band in the strong-coupling limit", is
correct but does not say this.

One caveat to state yourself before anyone else does. The angle dependence is not settled: at 4 E_R and
Ω = 1 E_R the band keeps 93 % of its width at m = 8 but only 59 % at m = 12, so at shallow depth the
threshold does move with angle, and the experiment's m = 22 is the first thing to compute next.

Why this matters for the graphene analogy: in graphene the electrons sit at Dirac points, the coupling varies
with the moiré pattern, and at the magic angle tunnelling paths interfere destructively while the states stay
spread over the whole moiré cell. Here bosons sit at k = 0, the coupling is uniform, and the flat band is an
atomic-limit band. Both are called flat bands; their physics is opposite. Honeycomb spin-twisted lattices are
a different case again, because there a uniform coupling does give magic flat bands (Luo and Zhang; Sui et
al.). Our claim is about the square bilayer only.

## 3. What the experiment actually measured

Meng et al. (Nature 615, 231, 2023) have three kinds of data.

Microwave spectroscopy: sweeping the frequency transfers atoms from the s-band of lattice 1 into the bands of
lattice 2, which calibrates the lattices and the coupling (0.243 E_R at V₀ = 4 E_R).

Images of the |2⟩ atoms in the trap: a density pattern with the 4.35 µm moiré period, which the microscope
resolves although the 395 nm lattice is not. These are compared with Gross–Pitaevskii ground states.

Momentum distributions after 18 ms of free expansion: peaks at the primary reciprocal lattice vectors, as for
any lattice superfluid, and much closer satellite peaks at the moiré momenta, one eleventh of the primary
spacing. Sharp moiré peaks mean phase coherence over distances beyond one moiré period. From the visibility of
the two families of peaks the paper builds a phase diagram in the plane of depth and coupling, with four
regions. SF shows both families: a superfluid coherent at all scales. SF-II keeps the primary peaks and loses
the moiré ones; the paper calls it "superfluid only with short-range coherence" and, in the Extended Data,
"superfluid domains embedded in a gapped insulator". MI is a Mott insulator with uniform integer filling and
no peaks. I is an insulator whose integer filling is modulated at the moiré scale.

Two experimental paths matter for you. At fixed coupling Ω_R = 0.24 E_R, deepening the lattice takes the
system SF → SF-II → MI: the moiré peaks vanish first, then the primary ones. At a fixed deep lattice (8 or
14 E_R), raising the coupling takes it MI → SF-II → SF, and the moiré peaks appear above Ω_R ≈ 0.5 E_R. This is
what "a modified superfluid-to-insulator transition" means: the critical depth depends on Ω_R. Coherence
survives holds of 40 ms. Not stated anywhere: the temperature, the radial trap frequency, the atom number in
the pancake. Only one angle was ever run, and no second twisted-lattice experiment exists.

Where our calculation already touches the data. At 4 E_R the whole measured coupling range, Ω_R ≤ 1 E_R, is
Ω/J ≤ 12, where the band still has 93 % of its width at m = 8: the moiré data were taken in the dispersive
regime, and the moiré pattern there is a density modulation of an ordinary superfluid. At 8 E_R the threshold
of 0.5 E_R is Ω/J ≈ 16, exactly where our band splits off and has lost three quarters of its width. Treat this
as a lead, not a result: it compares an interacting, trapped measurement at m = 22 with non-interacting
bands at m = 8.

## 4. What "superfluidity" means here and how a theorist quantifies it

You should be able to define each observable in the proposal in one breath.

Superfluid fraction. Move the confining potential slowly; the part of the gas that does not follow is
superfluid. In a calculation one imposes a phase twist and reads the fraction off the energy cost,
E(v) = E(0) + f_s N m v²/2. In a two-dimensional lattice it is a tensor. For a condensate in a single Bloch
state it equals the inverse effective mass of that band, f_s = m/m*; that identification combines a result of
Taylor and Zaremba with the sound relation below, and the m/m* we computed is a single-particle number, an
estimate at best once interactions smooth the density.

Leggett's bound. Superflow along x has to pass through every column of the density, and columns where the
density is low choke it. For a one-dimensional modulation f_s ≤ [⟨n⟩⟨1/n⟩]⁻¹, and for a weakly interacting
condensate this is an equality. In two dimensions there is an upper and a lower bracket (Pérez-Cruz,
Astrakharchik and Massignan), tight for square potentials. This is the rigorous reason a pocket array has
almost no superfluid fraction: the density between pockets goes to zero.

Sound velocities. With c_x along and c_y across a one-dimensional modulation, f_s = c_x²/c_y². This is how
the superfluid fraction has been measured since 2023: the Dalibard group with a modulated two-dimensional
gas (Leggett's identity confirmed) and then a triangular lattice (the full tensor, matching Gross–Pitaevskii),
Spielman with Bragg phonons in a lattice ("the speed of sound saturates Leggett's bound"), the Florence group
through Josephson oscillations in a supersolid. None of these is a moiré or Rabi-coupled gas. They are the
template, and they give you a measurable prediction.

Bogoliubov spectrum and depletion. Linearise the two-component Gross–Pitaevskii equation around the
condensate: excitation energies, the sound slope, the fraction of atoms outside the condensate even at zero
temperature, and a second, gapped branch, the relative-phase mode, whose gap is set by Ω and is therefore
itself modulated at the moiré scale.

Quantum geometry, and where the argument really stands. In a flat band the usual stiffness, proportional to
1/m*, vanishes; Peotta and Törmä showed that pairing still produces a superfluid weight, linear in the
attraction and set by the quantum metric of the Bloch states. Julku, Bruun and Törmä (2021) showed the
bosonic analogue on the kagome lattice: the sound velocity becomes linear in U and the depletion stays finite
as U → 0. Huhtinen et al. (2026) sharpened it: what counts is the condensate metric at the condensation
momentum, not an integral over the Brillouin zone, and a single flat band at a time-reversal-invariant
momentum with uniform density cannot be superfluid at all; a stable flat-band superfluid in two dimensions
needs at least three bands.

Be precise about how this excludes geometry for the pockets, because a careful listener will notice. The
uniform-density theorem does not apply to them: their density is anything but uniform. What applies is the
atomic limit. The review of Törmä, Peotta and Bernevig puts it plainly: a trivial flat band is an insulator
in the atomic limit, and there no current is possible. The minimal-metric result of Huhtinen et al. (2022)
and the Wannier-spread bound of Verma, Hazra and Randeria say the same in formulas, and our pockets have a
spread of about 2 % of the moiré cell. Nobody has computed the condensate metric of these bands; the
conclusion rests on the atomic-limit argument. One more result points the same way: for a two-band
condensate the superfluid density vanishes for U(2)-symmetric interactions (Zhang, preprint), and the
experiment is modelled with exactly such interactions. That has not been tested for pockets either.

## 5. Why theory is needed: the gap you are filling

State it plainly. The Nature paper's own theory is single-particle bands at three couplings, zero-temperature
Gross–Pitaevskii images, and a single-site Gutzwiller model in which the second lattice enters as an on-site
potential whose relation to Ω_R is not stated. The two superfluids and the shift of the transition are named,
not explained. Since then, Paul, Recher and Santos (2025) have done cluster-Gutzwiller ground states of the
two-layer lattice model and found Mott-like density waves and Bose-glass-like "mobility islands", with no
excitations, no superfluid fraction (percolation stands in for it), no temperature, no dynamics. Zeng, Zhu
and He did the case without a microwave. Johnstone et al. (2024) did exact quantum Monte Carlo for a
single-component summed twisted potential and found superfluid regions between pockets so thin, with
tunnelling between 10⁻¹¹ and 10⁻³ E_R, that thermal broadening at nanokelvin temperatures depletes them. Yang
et al. (2026) did fermions in honeycomb lattices. For the
Rabi-coupled condensate nobody has computed Bogoliubov spectra, sound or the superfluid-fraction tensor, its
bosonic quantum geometry, finite temperature, ramps of Ω, or an independent trapped simulation against the
data. No data set has been released.

So the need is threefold. Physically: nobody knows what happens to superfluidity in the window Ω/J ≈ 5–25,
where the band narrows by a factor of two to thirty while the density is still spread out, and where the
experiment sits. There the conventional 1/m* term, the geometric term (which rises to its maximum right there
and then dies), the Leggett suppression and the Josephson tunnelling between pockets all compete.
Experimentally: the measured phase diagram has no quantitative theory, and the thesis can hand back testable
numbers, the superfluid fraction and sound anisotropy versus Ω, the temperature at which moiré coherence
disappears, the way the threshold moves with depth and detuning. Methodologically: the lattice models in use
take their interlayer couplings from Gaussian overlaps, which reproduce the exact bandwidth well below
Ω/J ≈ 3 but are 30 % low at Ω/J ≈ 10 and ten times too small at Ω/J ≈ 32. Couplings must come from the bands.

The working hypothesis, and how it can fail. SF-II is a Josephson array of AA pockets: coherent pockets whose
mutual coupling has been lost, sitting in an insulating AB matrix. (The dipolar twisted bilayer at 5.21°,
studied by quantum Monte Carlo, shows the inverse texture, insulating islands in a superfluid.) It fails if the
superfluid fraction comes out above what the pocket picture allows, which is below Leggett's bound; if the
sound velocity turns out linear in U, the fingerprint of a geometric flat band, rather than growing as the
square root of U with a renormalised band mass; if the visibility threshold does not track Ω/J when the depth
changes, since at 14 E_R the single-particle collapse would sit near 0.15–0.2 E_R rather than 0.5; or if the
tunnelling between pockets lies far below any plausible temperature, in which case SF-II is a finite-size or
interaction-restored effect and the thesis turns into a temperature map.

## 6. What would be required to do it

Tools. The band code exists and is gate-checked: plane waves on the commensurate supercell, sparse
eigensolvers, seconds per parameter point at m = 8, a couple of minutes at m = 12. The next piece is a
two-component Gross–Pitaevskii solver with the Rabi term, imaginary time, split-step real time and twisted
boundary conditions for the superfluid fraction; this is an extension of the DFT solver you already have, and
complex fields, a second component and time stepping are the new parts (no open code offers this model
ready-made). Then Bogoliubov–de Gennes on the same supercell. Then a moiré Bose–Hubbard layer: Wannier
functions of the pockets, their tunnelling and interaction from the bands, cluster Gutzwiller, and an export
to the open worm-algorithm code of Sadoune and Pollet, which is exact for bosons. Finally a classical-field
layer (stochastic projected GPE, truncated Wigner) for temperature and ramps, valid only where modes are
fewer than particles and highly occupied; the Mott corner goes to Monte Carlo.

Compute. Everything except ensembles and Monte Carlo fits your 6 GB card. Monte Carlo needs a modest cluster
allocation; ask about VSC. The experiment's angle costs an estimated ten to twenty minutes per band point.

Skills to have ready. Bloch theory on supercells and zone folding; Wannier functions and why tunnelling falls
exponentially with depth; two-component Gross–Pitaevskii and Bogoliubov theory, density and spin branches;
Leggett's argument; the classical-field method and its validity conditions; the worm algorithm and the
winding-number estimator of stiffness; how time-of-flight images relate to momentum distributions.

Data you need from the experiment. The radial trap frequency, the atom number in the pancake and the
temperature are not in the paper, and neither are the scattering lengths of this particular pair of states.
Ask the Shanxi and Chicago authors.

Milestones. The old kill-check, whether a flat band with a large quantum metric exists, is already answered:
it does not, so nothing hangs on it any more. The decision point is now the figure of superfluid fraction
against Ω on the supercell, with Leggett's brackets, around week fourteen. If no geometric contribution ever
dominates, the thesis is the theory of the pocket crossover plus the comparison with experiment. If a window
appears, it becomes the centrepiece. Timeline: reading and parameter hunt (two weeks), solver extension (four),
bands at the experiment's angle and the pocket model (three), supercell superfluidity (five), trapped
experiment and lattice model (four), temperature and ramps (four), writing (four).

## 7. Questions you will be asked, and short answers

Why is there no magic angle here? Because the coupling is uniform in space and the bosons sit at k = 0. A
uniform coupling between two Bravais lattices cannot produce interference flat bands; it produces a moiré
potential that grows with Ω, and our bands flatten by localisation with a threshold in Ω/J, not in the angle.

Then why not switch to honeycomb lattices, where a uniform coupling does give magic flat bands? Because the
only data are on square lattices, and because the square case isolates the question the thesis asks: what
carries superfluidity when a band flattens without geometry to help. Honeycomb is the natural follow-up.

Isn't this what Paul, Recher and Santos did? They did zero-temperature ground states of a lattice model, with
percolation as the criterion for superfluidity and interlayer couplings that are ten times too small in the
collapsed regime. We compute excitations, the superfluid fraction, temperature, dynamics and the trap, in the
continuum where the lattice model is unreliable at 4 E_R, with couplings taken from the exact bands.

Why not just trust Gross–Pitaevskii? Because it is exact enough for the dispersive and crossover regimes at
these densities, and it is where the superfluid fraction and the sound come from; but it cannot describe the
Mott insulator or the thermal loss of coherence between weakly coupled pockets. Those go to the lattice model,
with Monte Carlo as the exact anchor.

Why bosons, when the graphene story is about fermions? Because the experiment is bosonic; because bosonic
flat-band superfluidity is the harder, less understood case; and because the superfluid fraction of a Bose
gas can be measured directly through sound.

What are the two superfluids in your picture? SF is coherence within and between pockets, both families of
peaks visible. SF-II is coherent pockets whose mutual Josephson coupling is lost or thermally washed out
inside an insulating matrix, so only the primary peaks survive. A hypothesis, with four stated ways to fail.

Does your threshold claim hold at the experiment's angle? Not shown yet. At 4 E_R the band width ratio drops
from 0.93 at m = 8 to 0.59 at m = 12; m = 22 is the first new computation, and the 8 E_R coincidence is a
lead, not a result.

What if the energy scales are too small, as the Monte Carlo suggests? Then the coherent pocket regime is
thermally out of reach at the current temperature, and the thesis delivers the map of what is reachable, with
a recommendation: shallower lattices, larger angles, lower temperature.

Where does machine learning come in? At the edges: a differentiable band code to search angle, depth,
coupling and detuning for windows that flatten without localising; surrogates for parameter scans; inference
of Ω, temperature and interaction strength from images. Nothing in the physics chapters depends on it.

Why the commensurate stand-in angle? Bloch's theorem needs periodicity; 2 arctan(1/22) differs from 5.21° by
less than the stated uncertainty, and the paper itself uses it. Incommensurability matters for single
particles, but interactions mask it at these densities.

Why keep the quantum metric at all? As a diagnostic. Computing the condensate metric along the crossover
shows where a geometric contribution peaks and why it dies, and it covers the two places where geometry
could still enter: band touchings, and a detuned coupling, which maps the single-particle problem onto a
Hofstadter model.

Is six months realistic? The literature is digested and checked, the novelty check is done, the band code is
written and the first result exists; the remaining code extends a solver that already passes its gates. The
risks are known, and each has a fallback.

## 8. Numbers to have in your head

Lattice: a = 395 nm, E_R = h × 3.67 kHz, θ = 5.21°, moiré period 11 a = 4.35 µm, stand-in 2 arctan(1/22).
Experiment: V₀ = 4 E_R for the moiré data (U/t = 1.67), up to 24 E_R for the transitions; Ω_R = 0.243 E_R for
the images, up to about 1 E_R; moiré peaks above 0.5 E_R at 8 or 14 E_R; holds of 40 ms; ω_z ≈ 2π × 1 kHz;
up to 5 × 10⁵ atoms; temperature not stated. Bands: exact J = 0.086, 0.031, 0.012 E_R at 4, 8, 12 E_R (the
deep-lattice formula is 36, 22, 16 % too high). Flattening at m = 8: half width at Ω/J ≈ 12 (8 E_R), 9
(12 E_R), 35 (4 E_R); band splits off at Ω/J ≈ 10–16; width 0.25 at 16, 0.004 at 32; Wannier spread 2 % of
the moiré cell. Angle caveat: 0.93 at m = 8 against 0.59 at m = 12 (4 E_R, Ω = 1 E_R). Gaussian-overlap
couplings: fine up to Ω/J ≈ 3, ten times off at 32. Tunnelling between pockets in the Monte Carlo:
10⁻¹¹–10⁻³ E_R at k_BT = 0.01–0.04 E_R.
