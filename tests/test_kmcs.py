import os
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import KMCS
from analyze_npz_3d import derive_occ_top, load_npz
from analyze_pillar_density import analyze_snapshot_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SNAPSHOT = REPOSITORY_ROOT / "examples" / "snapshot_final.npz"


class KMCSSmokeTests(unittest.TestCase):
    def test_version_and_random_generator_are_explicit(self):
        self.assertEqual(KMCS.__version__, "1.0.0")
        first = KMCS.make_rng(42)
        second = KMCS.make_rng(42)
        self.assertIsInstance(first.bit_generator, np.random.PCG64)
        np.testing.assert_array_equal(
            first.integers(0, 1000, size=12),
            second.integers(0, 1000, size=12),
        )

    def test_deposition_row_parsing(self):
        row = np.array([873.15, 2, 50, 0.04, 2, 1, 3, 2, 0, 0])
        temperature, frequency, pulses, flux, species, ratios = KMCS.parse_dep_row(row)
        self.assertAlmostEqual(temperature, 873.15)
        self.assertEqual(frequency, 2.0)
        self.assertEqual(pulses, 50)
        self.assertEqual(flux, 0.04)
        self.assertEqual(species, (2, 3))
        np.testing.assert_allclose(ratios, (1 / 3, 2 / 3))

    def test_substrate_height_matches_occupancy(self):
        occupancy, height = KMCS.init_state_with_steps(8, 8, 8, W=1)
        yy, xx = np.indices(height.shape)
        self.assertEqual(occupancy.shape, (8, 8, 8))
        self.assertTrue(np.all(occupancy[yy, xx, height] != 0))
        for y, x in np.ndindex(height.shape):
            occupied = np.flatnonzero(occupancy[y, x])
            self.assertEqual(height[y, x], occupied[-1])

    def test_deposition_is_repeatable_for_a_seed(self):
        occ_a, height_a = KMCS.init_state_with_steps(8, 8, 8, W=0)
        occ_b, height_b = KMCS.init_state_with_steps(8, 8, 8, W=0)
        KMCS.deposit_pulse(occ_a, height_a, (2, 3), (0.25, 0.75), 0.25, KMCS.make_rng(7))
        KMCS.deposit_pulse(occ_b, height_b, (2, 3), (0.25, 0.75), 0.25, KMCS.make_rng(7))
        np.testing.assert_array_equal(occ_a, occ_b)
        np.testing.assert_array_equal(height_a, height_b)

    def test_topview_is_a_bounded_rgb_image(self):
        occupancy, height = KMCS.init_state_with_steps(6, 5, 6, W=0)
        image = KMCS.topview_rgb(occupancy, height, {1: (0.2, 0.4, 0.6)})
        self.assertEqual(image.shape, (5, 6, 3))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_included_snapshot_can_be_analyzed(self):
        data = load_npz(EXAMPLE_SNAPSHOT)
        top_species = derive_occ_top(data)
        self.assertEqual(top_species.ndim, 2)
        analyzed_top, labels, metrics = analyze_snapshot_file(
            EXAMPLE_SNAPSHOT, species=2, min_area=3, connectivity=8
        )
        self.assertEqual(analyzed_top.shape, top_species.shape)
        self.assertEqual(labels.shape, top_species.shape)
        self.assertGreaterEqual(metrics["pillar_count_filtered"], 0)
        self.assertGreater(metrics["area_nm2"], 0)


if __name__ == "__main__":
    unittest.main()
