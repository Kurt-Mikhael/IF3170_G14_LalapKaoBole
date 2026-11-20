import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import classification_cost, cost_sensitive_threshold, predict_cost_sensitive


class CostSensitiveTests(unittest.TestCase):
    def test_threshold_reflects_error_costs(self):
        self.assertAlmostEqual(cost_sensitive_threshold(2, 1), 2 / 3)
        self.assertAlmostEqual(cost_sensitive_threshold(1, 2), 1 / 3)

    def test_predictions_use_threshold(self):
        probabilities = np.array([0.0, 0.5, 2 / 3, 0.9, 1.0])
        np.testing.assert_array_equal(
            predict_cost_sensitive(probabilities, false_positive_cost=2, false_negative_cost=1),[0, 0, 1, 1, 1],
        )

    def test_cost_counts_asymmetric_errors(self):
        report = classification_cost([0, 0, 1, 1], [1, 0, 0, 1], 2, 1)
        self.assertEqual(report, {"false_positive": 1, "false_negative": 1, "total_cost": 3.0})

    def test_invalid_inputs_are_rejected(self):
        for costs in ((0, 1), (1, -1), (np.nan, 1), (True, 1)):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                cost_sensitive_threshold(*costs)
        with self.assertRaises(ValueError):
            predict_cost_sensitive([0.2, 1.2])
        with self.assertRaises(ValueError):
            classification_cost([0, 2], [0, 1])


if __name__ == "__main__":
    unittest.main()
