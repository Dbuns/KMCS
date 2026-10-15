"""Regression checks for audited KMC behavior, not paper-level validation."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import contextlib
import io

import numpy as np
import KMCS as k

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

class KMCRegressionTests(unittest.TestCase):
    def test_all_six_face_moves_use_cubic_rules(self):
        neighborhood = np.zeros((3, 3, 3), dtype=np.uint8)
        neighborhood[:, :, 0] = 1
        for move in k.FACE_OFFSETS_6:
            with self.subTest(move=move):
                self.assertEqual(k.move_family(*move), "cubic")
                self.assertEqual(k.legacy_direction_weight(873.15, *move), 1.)
        for move in [(0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]:
            self.assertTrue(k.destination_allowed_by_family(neighborhood, *move))
        diagonal = [m for m in k.MOVE_DISPLACEMENTS_18 if m not in k.FACE_OFFSETS_6]
        self.assertEqual(len(diagonal), 12)
        for move in diagonal:
            self.assertNotEqual(k.move_family(*move), "cubic")

    def test_arrhenius_rate_uses_paper_table_1(self):
        self.assertEqual(k.k0, 1e12)
        self.assertEqual(k.Estatic, 1.)
        self.assertEqual(k.E_ES, .15)
        energy = k.default_run_config()["E_vals"]
        np.testing.assert_allclose(energy[:3, :3],
            [[0, .25, .49], [.25, .51, .25], [.49, .25, .51]])
        occ, h = k.init_state_with_steps(6, 6, 6, W=0)
        occ[3, 3, 1] = 2
        h[3, 3] = 1
        active = np.array([[3, 3, 1, 2]], dtype=np.int32)
        expected = k.k0 * np.exp(-(1. + float(energy[1, 0])) / (k.kB * 873.15))
        self.assertAlmostEqual(k.compute_rate_for_active(occ, energy, active, 0, 873.15), expected)

    def test_event_after_deadline_is_not_applied(self):
        occ, h = k.init_state_with_steps(4, 4, 4, W=0)
        active = np.array([[1, 1, 1, 2]], dtype=np.int32)
        occ[1, 1, 1] = 2
        h[1, 1] = 1
        class FixedRNG:
            def random(self):
                return .5
        before = occ.copy()
        with patch.object(k, "compute_rates_active", return_value=(np.array([1.]), [None])), \
                patch.object(k, "try_hop_cubic") as hop:
            result = k.relax_after_pulse_step3a(
                occ, h, np.ones((3, 3)), active, 873.15, FixedRNG(), .001)
        hop.assert_not_called()
        np.testing.assert_array_equal(occ, before)
        self.assertEqual(result[2:], ("time_limit", .001, 0))

    def test_no_events_still_reaches_pulse_deadline(self):
        occ, h = k.init_state_with_steps(4, 4, 4, W=0)
        result = k.relax_after_pulse_step3a(occ, h, np.ones((3, 3)),
            np.empty((0, 4), dtype=np.int32), 873.15, k.make_rng(42), .5)
        self.assertEqual(result[2:], ("no_events", .5, 0))

    def test_overflow_is_reported_before_state_changes(self):
        occ, h = k.init_state_with_steps(2, 2, 2, W=0)
        before, old_h = occ.copy(), h.copy()
        with self.assertRaisesRegex(ValueError, "increase Lz"):
            k.deposit_pulse(occ, h, (2,), (1.,), 10., k.make_rng(42))
        np.testing.assert_array_equal(occ, before)
        np.testing.assert_array_equal(h, old_h)

    def test_zero_flux_needs_no_species_and_draws_no_random_numbers(self):
        occ, h = k.init_state_with_steps(4, 4, 4, W=0)
        rng = k.make_rng(42)
        before = rng.bit_generator.state
        self.assertEqual(k.deposit_pulse(occ, h, (), (), 0., rng), (None,) * 4)
        self.assertEqual(rng.bit_generator.state, before)

    def test_two_site_dependency_is_invalidated(self):
        occ, _ = k.init_state_with_steps(8, 8, 8, W=0)
        active = np.array([[4, 4, 2, 2], [4, 3, 2, 2], [4, 2, 2, 2]])
        for y, x, z, s in active:
            occ[y, x, z] = s
        sites = k.build_active_site_set(active, occ)
        self.assertIsNone(k.source_rejection_reason(occ, active, 4, 4, 2, sites))
        occ[4, 2, 2] = 0
        sites.remove((4, 2, 2))
        self.assertEqual(k.source_rejection_reason(occ, active, 4, 4, 2, sites), "single_floater")
        affected = k.affected_active_indices(k.build_active_index_map(active, occ), 4, 2, 2, 8, 8, 8)
        self.assertIn(0, affected)

    def test_incremental_rates_match_full_recomputation_and_conserve_species(self):
        energy = k.default_run_config()["E_vals"]
        for steps in [0, 1, 2]:
            rng = k.make_rng(42 + steps)
            occ, h = k.init_state_with_steps(8, 8, 12, W=steps)
            ly, lx, lz = occ.shape
            xs, ys, ss, hb = k.deposit_pulse(occ, h, (2, 3), (.5, .5), .5, rng)
            active = k.build_active_list_from_last_deposit(xs, ys, ss, hb, occ.shape[2])
            sites = k.build_active_site_set(active, occ)
            index = k.build_active_index_map(active, occ)
            rates, geoms = k.compute_rates_active(occ, energy, active, 973.15, sites, True)
            counts = np.bincount(occ.ravel(), minlength=4)
            accepted = 0
            for _ in range(120):
                idx = k.pick_index_weighted(rng, rates)
                if idx is None:
                    break
                old = tuple(active[idx, :3])
                if k.try_hop_cubic(occ, h, energy, active, idx, 973.15, rng,
                                  active_sites=sites, hop_geometry=geoms[idx]):
                    accepted += 1
                    new = tuple(active[idx, :3])
                    affected = k.affected_active_indices(index, *old, ly, lx, lz)
                    index.pop(old)
                    index[new] = idx
                    affected |= k.affected_active_indices(index, *new, ly, lx, lz)
                    for j in affected:
                        geoms[j] = k.build_hop_geometry(occ, *active[j, :3])
                        rates[j] = k.compute_rate_for_active(occ, energy, active, j, 973.15, sites, geoms[j])
                full = k.compute_rates_active(occ, energy, active, 973.15, sites)
                np.testing.assert_allclose(rates, full, rtol=1e-12, atol=1e-15)
                np.testing.assert_array_equal(np.bincount(occ.ravel(), minlength=4), counts)
                expected_h = np.max(np.where(occ != 0, np.arange(occ.shape[2]), 0), axis=2)
                np.testing.assert_array_equal(h, expected_h)
                self.assertEqual(sites, set(map(tuple, active[:, :3])))
            self.assertGreater(accepted, 0)

    def test_bad_inputs_are_rejected_before_output_creation(self):
        config = k.default_run_config()
        for name, value in [("Lx", 0), ("Ly", 2.5), ("seed", -1), ("max_events_per_pulse", 0)]:
            with self.subTest(name=name), patch.object(k, "make_run_dir") as make_dir:
                invalid = dict(config, **{name: value})
                with self.assertRaises(ValueError):
                    k.run_single_kmcs(invalid)
                make_dir.assert_not_called()
        invalid_rows = [
            [0, 2, 1, .04, 2, 1], [873, 0, 1, .04, 2, 1],
            [873, 2, 1.5, .04, 2, 1], [873, 2, 1, -.1, 2, 1],
            [873, 2, 1, .04, 2], [873, 2, 1, .04, 2.5, 1],
            [873, 2, 1, .04, 256, 1], [873, 2, 1, .04, 2, -1, 3, 2],
            [873, 2, 1, .04, 0, 0, 2, 1], [873, 2, 1, .04],
            [np.nan, 2, 1, .04, 2, 1],
        ]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                k.parse_dep_row(np.array(row))
        with self.assertRaisesRegex(ValueError, "every deposited"):
            k.validate_run_inputs(config, np.array([[873, 2, 1, .04, 5, 1]]))

    def test_complete_small_multilayer_run_is_repeatable(self):
        results = []
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as out:
            for repeat in range(2):
                root = Path(out) / str(repeat)
                config = k.default_run_config()
                config.update(Lx=6, Ly=6, Lz=12, W=1,
                              max_events_per_pulse=10000, save_snapshots=False)
                # Short intervals keep this an uncapped plumbing check, not
                # a substitute for paper-scale morphology validation.
                config['DEP'] = np.array([
                    [873.15, 1e8, 2, .1, 2, .5, 3, .5],
                    [973.15, 1e8, 1, .1, 4, 1, 0, 0]])
                with patch.object(k, 'PROJECT_ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                    result = k.run_single_kmcs(config)
                self.assertEqual(result['validation_counters']['pulses_completed'], 3)
                self.assertEqual(result['validation_counters']['pulses_ended_by_max_events'], 0)
                with np.load(Path(result['run_dir']) / 'snapshot_final.npz') as data:
                    occ, height = data['Occ'].copy(), data['H'].copy()
                self.assertEqual(int(np.count_nonzero(occ == 4)), 4)
                self.assertEqual(int(np.count_nonzero((occ == 2) | (occ == 3))), 8)
                expected_h = np.max(np.where(occ != 0, np.arange(occ.shape[2]), 0), axis=2)
                np.testing.assert_array_equal(height, expected_h)
                results.append((occ, height, result['validation_counters']))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])
        self.assertEqual(results[0][2], results[1][2])

    def test_one_zero_flux_pulse_saves_final_state(self):
        config = k.default_run_config()
        config.update(Lx=4, Ly=4, Lz=6, W=0, save_snapshots=False)
        config['DEP'] = np.array([[873.15, 2, 1, 0, 0, 0]])
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as out:
            with patch.object(k, 'PROJECT_ROOT', Path(out)), contextlib.redirect_stdout(io.StringIO()):
                result = k.run_single_kmcs(config)
            with np.load(Path(result['run_dir']) / 'snapshot_final.npz') as data:
                self.assertEqual(float(data['simulation_time']), .5)
                self.assertEqual(int(np.count_nonzero(data['Occ'])), 16)


if __name__ == "__main__":
    unittest.main()
