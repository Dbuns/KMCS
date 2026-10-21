# Comparison with the published model

Status: **full-size temperature/frequency trend check passed** on 21 October
2026 for the corrected `1.1.0.dev0` simulator. All 25 built-in DOE conditions
(700–900 °C; 0.5/2/10/20/50 Hz) completed on a 128×128×35 lattice with 300
pulses and seed 42, without event-cap truncation. Final mass, heights, top
species and feature counts were independently checked from the saved states.
All five temperature cuts decrease and all five frequency cuts increase using
the predefined species-2, 8-connected, minimum-three-pixel analysis. Without
the area filter, the directions persist, with a tie at 900 °C between 0.5 and
2 Hz. At 20 Hz, density drops 85.9% from 700 to 900 °C; at 800 °C, it rises
9.71× from 0.5 to 50 Hz. No energies were retuned. This is a qualitative trend
check with one seed, not statistical equivalence or independent verification
of every 3D feature. The paper's 8 Hz point is kept distinct from the grid's
10 Hz point.

Compact checked data: `full_size_doe_2026-10-21.json`. Reproduce the updated
figures with `python validation/generate_full_size_figures.py`. The current
top-view montage additionally requires the original 25 saved snapshots. The
example NPZ is the corrected 900 °C / 2 Hz run; archived v1.0.0 examples remain
available in Git history and the DOI deposit. The fixed `DOE_plot.py` table and
the single-/three-material illustrations remain historical references.

## Earlier reduced-size screening

The **reduced-size mean trend-direction screening passed** for the corrected
`1.1.0.dev0` simulator. The previous public Python version (`c1b0590`) is the
comparison baseline; reproducing the original MATLAB implementation is not
required for this comparison. Software tests pass independently of scientific
acceptance. All 30 matched cases completed without event-cap truncation.
This earlier screen supported the intended directions of the mean temperature
and frequency response before the full-size check above. It is not by itself
full-size VAN morphology validation.

Reference: Cunha et al., *Morphology Evolution during Lithium-Based Vertically
Aligned Nanocomposite Growth*, https://doi.org/10.1021/acsami.9b15025.

## Model contract checked against the publication

- Six cubic and twelve diagonal candidate directions, subject to path and
  source-safety rules; periodic lateral boundaries.
- Only atoms deposited in the current pulse remain mobile; their positions
  freeze at the next pulse.
- `k0 = 1e12 /s`, static barrier 1.0 eV and ES barrier 0.15 eV.
- Face-sharing neighbors only, using the Table 1 material-pair contributions.
- Species mapping: 1 = STO substrate, 2 = LMO, 3 = LLTO.
- LMO/STO = 0.25, LLTO/STO = 0.49, LMO/LMO = LLTO/LLTO = 0.51,
  and LMO/LLTO = 0.25 eV. The paper explains that LMO/LMO and LMO/LLTO
  were inferred rather than measured directly.
- Deposition flux = 1/25 monolayer per pulse; LMO:LLTO = 1:2.
- The paper specifies an ES penalty for diagonal step-up/down moves. Both
  Python versions also weight in-plane diagonal moves; those weights are held
  fixed in this paired comparison. The direction-dispatch correction restores
  the four horizontal cubic directions.
- The paper describes checking time after hops, whereas the development version
  rejects sampled events beyond the pulse deadline. Compare the original
  timing convention explicitly when comparing the two Python versions.

## Matched Python trend screening

`compare_versions.py` runs both engines with identical inputs and seeds,
without tuning the energies. It saves run logs, final snapshots, per-case
metrics, a density-curve comparison and a morphology montage. Source hashes
are recorded, and completed cases can be resumed in the same output folder.

From the repository root:

```bash
python validation/compare_versions.py --grid 16 --pulses 25 \
  --seeds 7 42 159 --temperatures 700 800 900 --frequencies 10 20 50 \
  --output runs/comparison
python validation/compare_references.py --results runs/comparison
```

This evaluates five distinct conditions per seed and version: a temperature
sweep at 20 Hz and a frequency sweep at 800 °C, sharing the central point.
Both engines use W=1, 0.04 ML/pulse, LMO:LLTO=1:2, the default interaction
energies and an event cap of 100 million per pulse. The corrected version's
new direction, timing and cache logic remains active.

Feature counts use the same species-2, eight-connected, minimum-three-pixel
analysis for both versions. Raw counts are also recorded for sensitivity
checks. The plot shows mean density and sample standard deviation across seeds.
Check per-seed slopes and individual morphologies, not just a fitted mean line.

This is a reduced-size trend screening, not full-size paper validation.
Agreement means the temperature and frequency dependencies have the same
direction within the observed seed variability; identical pixel arrangements
or identical absolute densities are not required. Inconclusive results should
lead to longer/larger runs, not parameter tuning to force agreement.

## Completed screening result

The local 16×16, 25-pulse screen used three seeds and five distinct conditions
for each Python engine (30 total runs). Temperature means decrease at
700/800/900 °C at 20 Hz; frequency means are nondecreasing at 10/20/50 Hz at
800 °C. These directions agree with the experimental and original-paper
tables supplied by Daniel and with the corresponding cuts through the verified
Python DOE table. The historical table and experimental values were not used
to fit or tune either simulator.

For the primary three-pixel minimum-area analysis:

| Comparison | Published Python | Corrected Python |
|---|---:|---:|
| Density change, 700→900 °C at 20 Hz | −53% | −47% |
| Density change, 10→50 Hz at 800 °C | +30% | +36% |

All three corrected temperature seed slopes decrease; two of three frequency
seed slopes increase in both engines. One published temperature seed has equal
endpoint densities. The same mean directions remain when the minimum area is
reduced from three pixels to one. All final states retain their expected film
block count and have occupancy-consistent height maps.

The completed data represent only about one monolayer of early growth on a
small grid. A single component changes density by about 0.02568 nm^-2, and seed
variation is large relative to some between-condition differences. The result
does not establish a statistically precise frequency slope, equal absolute
densities, the whole DOE surface, or mature vertical pillar connectivity.

`reference_data.json` preserves the screenshot-transcribed paper/experimental
tables and their provenance. The two experimental central values differ
slightly and remain separate. `compare_references.py` reads the historical
Python DOE data at full precision from `DOE_plot.py` and produces normalized
reference curves, per-case integrity checks and a report. Raw mean curves and
seed SD remain available separately from the normalized plot.

The initial 32×32, 50-pulse pilot was stopped for runtime after eight completed
cases; its completed results remain separate. The 2 Hz corrected run took
about 100 minutes. Those results are not combined with the completed small-grid
screen. Larger grids, longer growth and fresh sub-10-Hz conditions remain
appropriate follow-up checks on the faster machine.

## Scientific comparison protocol

Use the Table 1 energies above without fitting them to the desired outcome.
Use 128×128 lateral grids and 300 pulses for the morphology comparison in
Figure 3. Choose Lz large enough to avoid any overflow, and record it.

1. Temperature sweep at 20 Hz: 700, 750, 800, 850 and 900 °C.
2. Frequency comparison at 800 °C: 0.5, 2, 8, 20 and 50 Hz, using the supplied
   original-paper and experimental table. The completed reduced-size screen
   covers only 10, 20 and 50 Hz; the lower-frequency region remains a fresh-run
   follow-up, despite being present in the historical reference tables.
3. Include the single-material wetting checks associated with Figure 2
   (12 pulses), once their precise temperature and substrate settings are
   confirmed from the original inputs.
4. Repeat across multiple seeds, retaining each result. Begin with seeds
   7, 42 and 159 and assess variability before deciding whether more are needed.
5. Use identical pillar analysis settings for new and old simulated snapshots:
   species, area cutoff, connectivity, pixel size and finite-image/periodic
   boundary treatment. The current analyzer uses finite-image components and
   a fixed 0.39 nm lattice spacing. Do not silently change that convention when
   comparing published curves.

The publication does not specify all initialization, segmentation and seed
settings. Preserve identical documented settings between the Python versions;
exact reproduction of the MATLAB implementation is outside this comparison.
Published numeric data, where available, can strengthen a later full-size
comparison. Those data cannot be reconstructed solely from a fitted DOE table.
The current stepped-substrate default W=1 is an implementation setting, not a
parameter established by the methods text. Compare identical initialization
across reference and corrected runs.

## Acceptance and reporting

- Every pulse must finish physically, without hitting the safety event cap;
  an event-capped run is incomplete and cannot pass validation.
- Confirm deposited and final species counts, height consistency and finite
  rates. Preserve seed, code revision, runtime versions and all parameters.
- Compare old and corrected simulators at identical inputs; inspect morphology
  and density curves with variation across seeds.
- At higher temperature expect larger features and lower number density;
  at higher frequency expect smaller features and higher number density.
  Inspect vertical connectivity, especially the low-frequency agglomerates.
- Compare with *simulated* reference densities separately from experimental
  densities. The paper reports a systematic scale difference between them
  and attributes it to the static barrier chosen for computational speed.
  Do not require equality to experimental density or retune energies to force it.
- Agree quantitative tolerances from reference data, uncertainty and repeated
  runs before judging results. A trend check alone is not exact reproduction.

The regression suite covers the implemented model contract and cache behavior;
it does not certify these scientific acceptance criteria. Publish corrected
figures and a new release only after recording the scientific comparison.
