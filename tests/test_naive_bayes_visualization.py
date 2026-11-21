import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.naive_bayes.visualization import (
    conditional_mutual_information,
    feature_relations,
    mutual_information,
    plot_naive_bayes_graph,
)

# 4000 baris: kelas bergantian 0/1, fitur acak yang tidak berkaitan dengan kelas
rng = np.random.default_rng(0)
LABELS = np.array([0, 1] * 2000)
RANDOM_BITS = rng.integers(0, 2, size=4000).astype(float)
OTHER_BITS = rng.integers(0, 2, size=4000).astype(float)


class NaiveBayesVisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_mutual_information_of_copy_of_class_is_one_bit(self):
        self.assertAlmostEqual(mutual_information(LABELS.astype(float), LABELS), 1.0, places=2)

    def test_mutual_information_of_unrelated_feature_is_near_zero(self):
        self.assertLess(mutual_information(RANDOM_BITS, LABELS), 0.01)

    def test_conditional_mutual_information_of_duplicate_features_is_one_bit(self):
        self.assertAlmostEqual(conditional_mutual_information(RANDOM_BITS, RANDOM_BITS, LABELS), 1.0, places=2)

    def test_conditional_mutual_information_of_independent_features_is_near_zero(self):
        self.assertLess(conditional_mutual_information(RANDOM_BITS, OTHER_BITS, LABELS), 0.01)

    def test_missing_values_are_ignored(self):
        with_nan = RANDOM_BITS.copy()
        with_nan[:100] = np.nan
        self.assertAlmostEqual(
            conditional_mutual_information(with_nan, with_nan, LABELS), 1.0, places=2
        )

    def test_feature_relations_sorts_pairs_from_strongest(self):
        X = np.column_stack([RANDOM_BITS, RANDOM_BITS, OTHER_BITS])
        class_scores, pair_scores = feature_relations(X, LABELS, ["a", "a_copy", "b"])
        self.assertEqual(set(class_scores), {"a", "a_copy", "b"})
        self.assertEqual(pair_scores[0][:2], ("a", "a_copy"))
        self.assertGreaterEqual(pair_scores[0][2], pair_scores[1][2])

    def test_plot_has_all_feature_names(self):
        X = np.column_stack([LABELS.astype(float), RANDOM_BITS, RANDOM_BITS, OTHER_BITS])
        names = ["salinan_kelas", "x", "x_copy", "y"]
        ax = plot_naive_bayes_graph(X, LABELS, names, class_name="batal", title="Graf uji")
        texts = {text.get_text() for text in ax.texts}
        self.assertTrue(set(names).issubset(texts))
        self.assertIn("batal", texts)
        self.assertEqual(ax.get_title(), "Graf uji")

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            feature_relations(np.zeros((4, 2)), [0, 1, 0, 1], ["only_one"])
        with self.assertRaises(ValueError):
            feature_relations(np.zeros((4, 2)), [0, 1, 0], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
