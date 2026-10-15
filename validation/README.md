# Comparison with the published model

Status: **pending** for the corrected `1.1.0.dev0` simulator. Software tests
pass independently of scientific acceptance. Historical figures, example NPZ
and the fixed table in `DOE_plot.py` represent earlier runs and must not be
presented as output of this corrected version.

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
- The paper specifies an ES penalty for diagonal step-up/down moves. The
  existing implementation also weights in-plane diagonal moves. This detail
  must be checked against Chris's original MATLAB code before declaring exact
  implementation equivalence; it was not changed speculatively in this cleanup.
- The paper describes checking time after hops, whereas the development version
  rejects sampled events beyond the pulse deadline. Compare the original
  timing convention explicitly when reproducing the reference implementation.

## Scientific comparison to run

Use the Table 1 energies above without fitting them to the desired outcome.
Use 128×128 lateral grids and 300 pulses for the morphology comparison in
Figure 3. Choose Lz large enough to avoid any overflow, and record it.

1. Temperature sweep at 20 Hz: 700, 750, 800, 850 and 900 °C.
2. Frequency comparison at 800 °C: include 0.5, 2 and 8 Hz for Figure 4;
   add the actual Figure 5 frequency points when the original numeric data
   or a documented digitization is available.
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
settings. Recover the original MATLAB code/input files and numeric data where
possible. These details cannot be reconstructed solely from a fitted DOE table.
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
