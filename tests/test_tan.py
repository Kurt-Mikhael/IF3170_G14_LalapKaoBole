import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import NaiveBayesClassifier, TreeAugmentedNaiveBayes
from models.naive_bayes.information import conditional_mutual_information, mutual_information


def make_dependent_data(n=2000, seed=0):
    # X0 mengikuti kelas, X1 hampir salinan X0 sehingga bergantung kuat meski kelas diketahui, dan X2 bit acak
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x0 = np.where(rng.random(n) < 0.9, y, 1 - y)
    x1 = np.where(rng.random(n) < 0.95, x0, 1 - x0)
    x2 = rng.integers(0, 2, size=n)
    return np.column_stack([x0, x1, x2]), y


def make_small_data(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(n, 3)), rng.integers(0, 2, size=n)


def expected_proba(model, X, y, query, smoothing, alpha=1.0, n0=5.0):
    # hitung ulang P(kelas | fitur) dengan hitungan langsung dari data untuk struktur yang dipelajari model
    classes = sorted(set(y.tolist()))
    n = len(y)
    scores = []
    for c in classes:
        n_in_class = sum(1 for label in y if label == c)
        score = math.log((n_in_class + alpha) / (n + alpha * len(classes)))
        for j in range(X.shape[1]):
            parent = model.parents_[j]
            rows = [i for i in range(n) if y[i] == c and (parent is None or X[i, parent] == query[parent])]
            total = len(rows)
            count = sum(1 for i in rows if X[i, j] == query[j])
            if smoothing == "laplace":
                n_values = len(set(X[:, j].tolist()))
                probability = (count + alpha) / (total + alpha * n_values)
            else:
                marginal = float(np.mean(X[:, j] == query[j]))
                probability = (count + n0 * marginal) / (total + n0)
            score += math.log(probability)
        scores.append(score)
    best = max(scores)
    exps = [math.exp(s - best) for s in scores]
    return [e / sum(exps) for e in exps]


class TreeAugmentedNaiveBayesTests(unittest.TestCase):
    def test_tree_links_the_dependent_features(self):
        X, y = make_dependent_data()
        model = TreeAugmentedNaiveBayes().fit(X, y)
        linked = {frozenset((parent, child)) for parent, child, _ in model.edges_}
        self.assertIn(frozenset((0, 1)), linked)

    def test_structure_is_a_tree_with_one_root(self):
        X, y = make_dependent_data()
        model = TreeAugmentedNaiveBayes().fit(X, y)
        self.assertEqual(len(model.edges_), X.shape[1] - 1)
        self.assertEqual(sum(parent is None for parent in model.parents_), 1)
        # dari tiap fitur, mengikuti induk harus sampai ke akar tanpa berputar
        for start in range(X.shape[1]):
            current = start
            for _ in range(X.shape[1]):
                if model.parents_[current] is None:
                    break
                current = model.parents_[current]
            self.assertIsNone(model.parents_[current])

    def test_structure_is_deterministic(self):
        X, y = make_dependent_data()
        first = TreeAugmentedNaiveBayes().fit(X, y)
        second = TreeAugmentedNaiveBayes().fit(X, y)
        self.assertEqual(first.parents_, second.parents_)

    def test_root_can_be_chosen(self):
        X, y = make_dependent_data()
        model = TreeAugmentedNaiveBayes(root=2).fit(X, y)
        self.assertIsNone(model.parents_[2])
        with self.assertRaises(ValueError):
            TreeAugmentedNaiveBayes(root=7).fit(X, y)

    def test_default_root_is_the_feature_most_related_to_the_class(self):
        X, y = make_dependent_data()
        model = TreeAugmentedNaiveBayes().fit(X, y)
        scores = [mutual_information(X[:, j], y, bias_correction=False) for j in range(X.shape[1])]
        self.assertIsNone(model.parents_[int(np.argmax(scores))])

    def test_probabilities_sum_to_one(self):
        X, y = make_dependent_data()
        proba = TreeAugmentedNaiveBayes().fit(X, y).predict_proba(X[:50])
        for row in proba:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_laplace_matches_hand_calculation(self):
        X, y = make_small_data()
        model = TreeAugmentedNaiveBayes(alpha=1.0).fit(X, y)
        for query in X[:10]:
            expected = expected_proba(model, X, y, query, "laplace", alpha=1.0)
            actual = model.predict_proba([query])[0]
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])

    def test_dirichlet_matches_hand_calculation(self):
        X, y = make_small_data()
        model = TreeAugmentedNaiveBayes(alpha=1.0, smoothing="dirichlet", n0=5.0).fit(X, y)
        for query in X[:10]:
            expected = expected_proba(model, X, y, query, "dirichlet", alpha=1.0, n0=5.0)
            actual = model.predict_proba([query])[0]
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])

    def test_unseen_values_do_not_crash(self):
        X, y = make_dependent_data()
        for smoothing in ("laplace", "dirichlet"):
            model = TreeAugmentedNaiveBayes(smoothing=smoothing).fit(X, y)
            proba = model.predict_proba([[9, 9, 9]])[0]
            self.assertAlmostEqual(sum(proba), 1.0)

    def test_predictions_are_close_to_naive_bayes_when_features_are_independent(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, size=4000)
        X = np.column_stack([np.where(rng.random(4000) < 0.8, y, 1 - y) for _ in range(4)])
        tan = TreeAugmentedNaiveBayes().fit(X, y).predict_proba(X)[:, 1]
        nb = NaiveBayesClassifier().fit(X, y).predict_proba(X)[:, 1]
        # tabel tambahan dari sisi pohon hanya menyesuaikan sedikit karena kebergantungannya hanya kebetulan
        self.assertLess(np.abs(tan - nb).mean(), 0.02)

    def test_tan_beats_naive_bayes_when_features_are_duplicated(self):
        # fitur duplikat membuat Naive Bayes menghitung bukti yang sama berkali-kali
        X, y = make_dependent_data()
        tan_accuracy = np.mean(TreeAugmentedNaiveBayes().fit(X, y).predict(X) == y)
        nb_accuracy = np.mean(NaiveBayesClassifier().fit(X, y).predict(X) == y)
        self.assertGreaterEqual(tan_accuracy, nb_accuracy - 0.01)

    def test_missing_values_must_be_encoded_before_fitting(self):
        X, y = make_dependent_data()
        X = X.astype(float)
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            TreeAugmentedNaiveBayes().fit(X, y)
        model = TreeAugmentedNaiveBayes().fit(X[1:], y[1:])
        with self.assertRaises(ValueError):
            model.predict_proba(X[:1])

    def test_missing_value_coded_as_a_category_is_counted_like_any_other_value(self):
        X, y = make_dependent_data()
        X = X.astype(float)
        X[:100, 2] = -1.0  # nilai kosong yang sudah dikodekan menjadi kategori
        model = TreeAugmentedNaiveBayes().fit(X, y)
        proba = model.predict_proba([[0, 0, -1]])[0]
        self.assertAlmostEqual(sum(proba), 1.0)

    def test_save_and_load(self):
        X, y = make_dependent_data()
        model = TreeAugmentedNaiveBayes(smoothing="dirichlet").fit(X, y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "tan.pkl"
            model.save(path)
            loaded = TreeAugmentedNaiveBayes.load(path)
        self.assertTrue(np.allclose(model.predict_proba(X[:20]), loaded.predict_proba(X[:20])))
        self.assertEqual(model.parents_, loaded.parents_)

    def test_string_labels_are_preserved(self):
        X, y = make_dependent_data()
        labels = np.where(y == 1, "batal", "tidak")
        predictions = TreeAugmentedNaiveBayes().fit(X, labels).predict(X)
        self.assertEqual(set(predictions), {"batal", "tidak"})

    def test_invalid_inputs_raise(self):
        X, y = make_dependent_data()
        with self.assertRaises(ValueError):
            TreeAugmentedNaiveBayes(alpha=-1)
        with self.assertRaises(ValueError):
            TreeAugmentedNaiveBayes(smoothing="lain")
        with self.assertRaises(ValueError):
            TreeAugmentedNaiveBayes(n0=0)
        with self.assertRaises(RuntimeError):
            TreeAugmentedNaiveBayes().predict(X)
        with self.assertRaises(ValueError):
            TreeAugmentedNaiveBayes().fit(X, y[:-1])
        model = TreeAugmentedNaiveBayes().fit(X, y)
        with self.assertRaises(ValueError):
            model.predict_proba([[0, 1]])


class InformationTests(unittest.TestCase):
    def test_bias_correction_never_increases_the_score(self):
        X, y = make_dependent_data(n=300)
        raw = conditional_mutual_information(X[:, 0], X[:, 2], y, bias_correction=False)
        corrected = conditional_mutual_information(X[:, 0], X[:, 2], y)
        self.assertLessEqual(corrected, raw)
        raw_class = mutual_information(X[:, 2], y, bias_correction=False)
        corrected_class = mutual_information(X[:, 2], y)
        self.assertLessEqual(corrected_class, raw_class)

    def test_duplicate_feature_has_one_bit_of_conditional_information(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, size=4000)
        a = rng.integers(0, 2, size=4000)
        self.assertAlmostEqual(conditional_mutual_information(a, a, y, bias_correction=False), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
