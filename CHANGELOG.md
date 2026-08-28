# Changelog

All notable changes to KMCS will be documented in this file.

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
