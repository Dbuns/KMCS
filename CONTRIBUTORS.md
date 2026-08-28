# Development history and contributors

KMCS has developed through three implementations of the same scientific model.

- **Bouwe Kuiper** developed the earlier C++ implementation during his PhD at
  the University of Twente (2009-2014). This implementation provided groundwork
  for the later MATLAB model.
- **Chris Vos** developed the MATLAB implementation during his master's thesis
  at the University of Twente. That implementation was used in the research
  reported in the related 2019 publication.
- **Daniel M. Cunha** developed and validated the present Python adaptation and
  its extensions, including multi-material support, automated parameter sweeps,
  Design of Experiments analysis, visualization, documentation, and public
  release preparation.

## Random-number implementations

The historical implementations used different versions of the Mersenne
Twister:

- Bouwe's C++ implementation used Mersenne Twister version 1.1 from the
  [original Hiroshima University distribution](https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html).
- Chris's MATLAB implementation used MATLAB's built-in Mersenne Twister.

The current Python implementation uses NumPy's PCG64 generator. It does not
contain source code from the historical C++ random-number implementation or
from MATLAB.

The Python port, optimization, documentation, and repository preparation were
carried out with substantial assistance from ChatGPT by OpenAI. The scientific
model, parameter choices, validation, interpretation, and release decisions
remain the responsibility of the human authors.
