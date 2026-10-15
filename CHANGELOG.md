# Changelog

All notable changes to KMCS will be documented in this file.

## [Unreleased] - 1.1.0.dev0

Local development version. Scientific comparison with the published model is
pending; v1.0.0 remains the archived reference release.

### Changed

- Removed unused early hopping implementations, helper functions, destination
  energy caches, ineffective plotting/profiling settings and unused DOE arrays.
- `plot_every` controls interactive plotting; `enable_profiling` profiles the
  complete run. Removed `enable_plotting` and `profile_pulses`.
- Added validation of deposition schedules, species IDs, energy-matrix shape,
  grid settings and output intervals before a run creates output files.
- Updated ignore rules for local virtual environments and environment files.

### Fixed

- Classify all six face-neighbor directions as cubic hops.
- Do not apply a hop whose sampled event time reaches or exceeds the next pulse.
- Refresh cached geometry/rates for all actual local dependencies, including
  occupancy two sites away, and resummate total rates after updates.
- Reject a pulse that would overflow the lattice before changing any occupancy.
- Advance no-event intervals to their pulse deadline; save snapshots for a
  single zero-flux pulse.

### Verification

- Added focused regressions for hopping directions, Arrhenius constants,
  deadline handling, cache/full-recompute equivalence, species conservation,
  height consistency, overflow and invalid inputs.
- Tests and small runs are not a replacement for the full paper comparison.
  See `validation/README.md` for the scientific acceptance procedure.

## [1.0.0] - 2026-08-28

First formal public research-software release.

### Added

- Python implementation of the validated lattice KMC model.
- Configurable single- and multi-material deposition schedules.
- Support for successive layers with independently defined growth parameters.
- Automated temperature-frequency Design of Experiments runs.
- Pillar number-density analysis.
- Two- and three-dimensional snapshot analysis and visualization.
- Curated example data and validation figures.
- Reproducible seeds and recorded software/runtime metadata.
- Apache-2.0 licence, citation metadata, contributor history, and smoke tests.

### Notes

- Materials 1-3 reproduce the model parameterization associated with the
  published LMO-LLTO study.
- Material 4 remains an unvalidated demonstration with placeholder interaction
  energies.
- The auxiliary roughness output remains experimental and is not part of the
  validated published workflow.
