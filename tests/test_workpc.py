"""Fast checks of the work-PC profile and completion safeguards."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import KMCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workpc"))
import run_full_doe as runner


class WorkPCChecks(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((ROOT / "workpc/full_doe_config.json").read_text(encoding="utf-8"))

    def test_full_profile_preserves_original_energies(self):
        original = copy.deepcopy(self.settings)
        config = runner.build_config(KMCS, self.settings, 800, 20, 42)
        KMCS.validate_run_inputs(config, config["DEP"])
        np.testing.assert_array_equal(config["E_vals"], KMCS.default_run_config()["E_vals"])
        self.assertEqual(self.settings, original)
        self.assertEqual(self.settings["grid"], [128, 128, 35])
        self.assertEqual(config["DEP"][0, 2], 300)
        self.assertEqual(len(self.settings["temperatures_C"]) * len(self.settings["frequencies_Hz"]), 25)

    def run_tiny_case(self, output, frequency, limit):
        settings = copy.deepcopy(self.settings)
        settings.update(grid=[8, 8, 12], pulses=2, max_events_per_pulse=limit)
        return runner.execute_case(dict(output=output, case_id="test", engine="corrected",
            T_C=800, f_Hz=frequency, seed=42, fingerprint="test", settings=settings,
            source=runner.load_engine_source("corrected")))

    def test_complete_case_has_portable_snapshot_and_conserved_mass(self):
        with tempfile.TemporaryDirectory() as output:
            record = self.run_tiny_case(output, 1e8, 10000)
            self.assertEqual(record["status"], "complete", record)
            self.assertEqual(record["film_count"], 6)
            self.assertTrue(record["integrity_ok"])
            self.assertFalse(Path(record["snapshot"]).is_absolute())
            self.assertTrue((Path(output) / record["snapshot"]).is_file())

    def test_event_capped_case_is_not_accepted_as_complete(self):
        with tempfile.TemporaryDirectory() as output:
            record = self.run_tiny_case(output, 0.5, 1)
            self.assertEqual(record["status"], "invalid", record)
            self.assertGreater(record["capped_pulses"], 0)

    def test_failed_case_records_error_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as output:
            record = runner.execute_case(dict(output=output, case_id="broken", engine="corrected",
                T_C=800, f_Hz=20, seed=42, fingerprint="test", settings=self.settings,
                source="raise RuntimeError('intentional test failure')"))
            self.assertEqual(record["status"], "failed")
            self.assertIn("intentional test failure", record["error"])
            self.assertTrue((Path(output) / record["attempt"] / "error.txt").is_file())


if __name__ == "__main__":
    unittest.main()
