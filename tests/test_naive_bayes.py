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

# Varian Play Tennis untuk contoh smoothing, tanpa baris "Hot" berlabel Yes dan tanpa baris "Overcast" berlabel No.
TENNIS2_X = [
    [0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [2, 1, 0, 0], [2, 2, 1, 0],
    [2, 2, 1, 1], [1, 2, 1, 1], [0, 1, 0, 0], [0, 2, 1, 0], [2, 1, 1, 0],
    [0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0], [2, 1, 0, 1],
]
TENNIS2_Y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

# Dua fitur kategorikal: channel (Online=0, Offline=1) dan permintaan khusus (tidak=0, ada=1).
# Label: tidak batal=0, batal=1.
HOTEL_X = [
    [0, 0], [0, 0], [0, 1], [0, 1], [0, 1],
    [1, 0], [1, 1], [1, 1], [1, 1], [1, 0],
]
HOTEL_Y = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

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
        # prior dan likelihood sama-sama di-smoothing, hanya fitur channel
        # P(kelas) = (hitungan + alpha) / (n + alpha * 2 kelas)
        # P(v | kelas) = (hitungan + alpha) / (total kelas + alpha * K)
        channel_only = [[row[0]] for row in HOTEL_X]
        model = NaiveBayesClassifier(alpha=1.0).fit(channel_only, HOTEL_Y)
        batal = (4 + 1) / (10 + 2) * (3 + 1) / (4 + 2)
        tidak = (6 + 1) / (10 + 2) * (2 + 1) / (6 + 2)
        proba = model.predict_proba([[0]])[0]
        self.assertAlmostEqual(proba[1], batal / (batal + tidak))

    def test_prior_is_smoothed_with_alpha(self):
        # 4 baris batal dan 6 baris tidak batal
        smoothed = NaiveBayesClassifier(alpha=1.0).fit(HOTEL_X, HOTEL_Y)
        self.assertAlmostEqual(math.exp(smoothed.class_log_prior_[0]), (6 + 1) / (10 + 2))
        self.assertAlmostEqual(math.exp(smoothed.class_log_prior_[1]), (4 + 1) / (10 + 2))
        # alpha = 0 kembali ke frekuensi biasa
        plain = NaiveBayesClassifier(alpha=0).fit(HOTEL_X, HOTEL_Y)
        self.assertAlmostEqual(math.exp(plain.class_log_prior_[0]), 6 / 10)
        self.assertAlmostEqual(math.exp(plain.class_log_prior_[1]), 4 / 10)

    def test_play_tennis_with_laplace_smoothing_on_prior_and_likelihood(self):
        # query <Overcast, Hot, High, Strong>, skor dihitung manual dengan alpha = 1
        model = NaiveBayesClassifier(alpha=1.0).fit(TENNIS2_X, TENNIS2_Y)
        yes = (9 + 1) / (14 + 2) * (4 + 1) / (9 + 3) * (0 + 1) / (9 + 3) * (3 + 1) / (9 + 2) * (3 + 1) / (9 + 2)
        no = (5 + 1) / (14 + 2) * (0 + 1) / (5 + 3) * (2 + 1) / (5 + 3) * (4 + 1) / (5 + 2) * (3 + 1) / (5 + 2)
        proba = model.predict_proba([[1, 0, 0, 1]])[0]
        self.assertAlmostEqual(proba[0], no / (no + yes))
        self.assertAlmostEqual(proba[1], yes / (no + yes))
        self.assertEqual(model.predict([[1, 0, 0, 1]])[0], 0)

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

    def test_missing_value_coded_as_its_own_category(self):
        # nilai kosong yang sudah dikodekan menjadi kategori (-1) dihitung seperti nilai lain
        X = [[float(value) for value in row] for row in HOTEL_X]
        X[0][1] = -1.0
        model = NaiveBayesClassifier(alpha=1.0).fit(X, HOTEL_Y)
        # permintaan khusus punya 3 nilai (-1, 0, 1), -1 hanya muncul sekali di kelas batal
        batal = (4 + 1) / (10 + 2) * (3 + 1) / (4 + 2) * (1 + 1) / (4 + 3)
        tidak = (6 + 1) / (10 + 2) * (2 + 1) / (6 + 2) * (0 + 1) / (6 + 3)
        proba = model.predict_proba([[0, -1]])[0]
        self.assertAlmostEqual(proba[1], batal / (batal + tidak))

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
            NaiveBayesClassifier().fit(TENNIS_X, TENNIS_Y[:-1])

        model = NaiveBayesClassifier().fit(TENNIS_X, TENNIS_Y)
        with self.assertRaises(ValueError):
            model.predict_proba([[0, 0]])
        with self.assertRaises(ValueError):
            model.predict_proba([[np.inf, 0, 0, 0]])

if __name__ == "__main__":
    unittest.main()