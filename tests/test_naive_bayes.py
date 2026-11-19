import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import NaiveBayesClassifier

# Dataset Play Tennis klasik.
# outlook: Sunny=0 Overcast=1 Rain=2 | temp: Hot=0 Mild=1 Cool=2
# humidity: High=0 Normal=1 | wind: Weak=0 Strong=1 | label: No=0 Yes=1
TENNIS_X = [
    [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [2, 1, 0, 0], [2, 2, 1, 0],
    [2, 2, 1, 1], [1, 2, 1, 1], [0, 1, 0, 0], [0, 2, 1, 0], [2, 1, 1, 0],
    [0, 1, 1, 1], [1, 1, 0, 1], [1, 0, 1, 0], [2, 1, 0, 1],
]
TENNIS_Y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

# Dua fitur kategorikal: channel (Online=0, Offline=1) dan permintaan khusus
# (tidak=0, ada=1). Label: tidak batal=0, batal=1.
HOTEL_X = [
    [0, 0], [0, 0], [0, 1], [0, 1], [0, 1],
    [1, 0], [1, 1], [1, 1], [1, 1], [1, 0],
]
HOTEL_Y = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

def gaussian_pdf(x, mean, var):
    return math.exp(-((x - mean) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)

class NaiveBayesTests(unittest.TestCase):
    def test_play_tennis_matches_hand_calculation(self):
        # query <Sunny, Cool, High, Strong>, skor dihitung manual tanpa smoothing
        model = NaiveBayesClassifier(alpha=0).fit(TENNIS_X, TENNIS_Y)
        yes = 9 / 14 * 2 / 9 * 3 / 9 * 3 / 9 * 3 / 9
        no = 5 / 14 * 3 / 5 * 1 / 5 * 4 / 5 * 3 / 5
        proba = model.predict_proba([[0, 2, 0, 1]])[0]
        self.assertAlmostEqual(proba[0], no / (no + yes))
        self.assertAlmostEqual(proba[1], yes / (no + yes))
        self.assertEqual(model.predict([[0, 2, 0, 1]])[0], 0)

    def test_two_features_match_hand_calculation(self):
        # query Online + ada permintaan, skor dihitung manual tanpa smoothing
        model = NaiveBayesClassifier(alpha=0).fit(HOTEL_X, HOTEL_Y)
        batal = 0.4 * 3 / 4 * 1 / 4
        tidak = 0.6 * 2 / 6 * 5 / 6
        proba = model.predict_proba([[0, 1]])[0]
        self.assertAlmostEqual(proba[1], batal / (batal + tidak))

    def test_zero_frequency_without_smoothing_kills_class(self):
        # P(overcast | no) = 0, sehingga kelas "no" bernilai 0 untuk overcast
        model = NaiveBayesClassifier(alpha=0).fit(TENNIS_X, TENNIS_Y)
        proba = model.predict_proba([[1, 2, 0, 1]])[0]
        self.assertEqual(proba[0], 0.0)
        self.assertEqual(proba[1], 1.0)

    def test_laplace_smoothing_matches_formula(self):
        # P(v | kelas) = (hitungan + alpha) / (total kelas + alpha * K), hanya fitur channel
        channel_only = [[row[0]] for row in HOTEL_X]
        model = NaiveBayesClassifier(alpha=1.0).fit(channel_only, HOTEL_Y)
        batal = 0.4 * (3 + 1) / (4 + 2)
        tidak = 0.6 * (2 + 1) / (6 + 2)
        proba = model.predict_proba([[0]])[0]
        self.assertAlmostEqual(proba[1], batal / (batal + tidak))

    def test_unseen_value_does_not_crash_with_smoothing(self):
        model = NaiveBayesClassifier(alpha=1.0).fit(HOTEL_X, HOTEL_Y)
        proba = model.predict_proba([[7, 1]])[0]
        self.assertAlmostEqual(sum(proba), 1.0)

    def test_missing_value_is_skipped(self):
        # fitur yang kosong sama saja dengan fitur itu tidak dipakai
        both = NaiveBayesClassifier(alpha=1.0).fit(HOTEL_X, HOTEL_Y)
        channel_only = [[row[0]] for row in HOTEL_X]
        first = NaiveBayesClassifier(alpha=1.0).fit(channel_only, HOTEL_Y)
        proba_both = both.predict_proba([[0, np.nan]])[0]
        proba_first = first.predict_proba([[0]])[0]
        self.assertAlmostEqual(proba_both[1], proba_first[1])

    def test_missing_values_in_training_are_not_counted(self):
        # baris pertama (batal) kehilangan permintaan khusus, jadi hanya 3 baris batal yang dihitung
        X = [[float(value) for value in row] for row in HOTEL_X]
        X[0][1] = np.nan
        model = NaiveBayesClassifier(alpha=0).fit(X, HOTEL_Y)
        batal = 0.4 * 3 / 4 * 1 / 3
        tidak = 0.6 * 2 / 6 * 5 / 6
        proba = model.predict_proba([[0, 1]])[0]
        self.assertAlmostEqual(proba[1], batal / (batal + tidak))

    def test_gaussian_feature_matches_hand_calculation(self):
        X = [[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]
        y = [0, 0, 0, 1, 1, 1]
        model = NaiveBayesClassifier(feature_types=["gaussian"], var_smoothing=0.0).fit(X, y)
        var = 2 / 3  # varians {1,2,3} dan {10,11,12}
        score0 = 0.5 * gaussian_pdf(4.0, 2.0, var)
        score1 = 0.5 * gaussian_pdf(4.0, 11.0, var)
        proba = model.predict_proba([[4.0]])[0]
        self.assertAlmostEqual(proba[0], score0 / (score0 + score1))
        self.assertAlmostEqual(proba[1], score1 / (score0 + score1))

    def test_mixed_feature_types(self):
        channel = [row[0] for row in HOTEL_X]
        numeric = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        X = np.column_stack([channel, numeric])
        model = NaiveBayesClassifier(feature_types=["categorical", "gaussian"])
        proba = model.fit(X, HOTEL_Y).predict_proba(X)
        for row in proba:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_predict_is_argmax_of_predict_proba(self):
        model = NaiveBayesClassifier().fit(TENNIS_X, TENNIS_Y)
        proba = model.predict_proba(TENNIS_X)
        for row, label in zip(proba, model.predict(TENNIS_X)):
            self.assertEqual(label, model.classes_[np.argmax(row)])

    def test_many_features_do_not_underflow(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 3, size=(200, 400))
        y = rng.integers(0, 2, size=200)
        proba = NaiveBayesClassifier().fit(X, y).predict_proba(X)
        for row in proba:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_string_labels_are_preserved(self):
        y = ["ya" if label == 1 else "tidak" for label in HOTEL_Y]
        predictions = NaiveBayesClassifier().fit(HOTEL_X, y).predict(HOTEL_X)
        self.assertEqual(set(predictions), {"tidak", "ya"})

    def test_save_and_load(self):
        model = NaiveBayesClassifier(alpha=0.5).fit(TENNIS_X, TENNIS_Y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "nb.pkl"
            model.save(path)
            loaded = NaiveBayesClassifier.load(path)
        before = model.predict_proba(TENNIS_X)
        after = loaded.predict_proba(TENNIS_X)
        self.assertTrue(np.allclose(before, after))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(RuntimeError):
            NaiveBayesClassifier().predict(TENNIS_X)
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(alpha=-1)
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(feature_types=["kategori"])
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().fit(TENNIS_X, TENNIS_Y[:-1])
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(feature_types=["categorical"]).fit(TENNIS_X, TENNIS_Y)

        model = NaiveBayesClassifier().fit(TENNIS_X, TENNIS_Y)
        with self.assertRaises(ValueError):
            model.predict_proba([[0, 0]])
        with self.assertRaises(ValueError):
            model.predict_proba([[np.inf, 0, 0, 0]])

if __name__ == "__main__":
    unittest.main()