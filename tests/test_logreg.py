import pickle
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models import LogisticRegressionClassifier


class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2., 1.], [-1., -1.], [1., -1.], [2., 1.]])
        self.y = np.array([0, 0, 1, 1])

    def test_gradients_match_finite_differences(self):
        for penalty in (None, 'l1', 'l2', 'elasticnet'):
            model = LogisticRegressionClassifier(penalty=penalty, reg_strength=.3)
            theta = np.array([.4, -.2, .1])  # L1 evaluated away from zero.
            loss = lambda t: model._loss_gradient(self.X, self.y, t[:2], t[2])[0]
            _, dw, db = model._loss_gradient(self.X, self.y, theta[:2], theta[2])
            numerical = []
            for i in range(3):
                delta = np.eye(3)[i] * 1e-6
                numerical.append((loss(theta + delta) - loss(theta - delta)) / 2e-6)
            np.testing.assert_allclose(np.r_[dw, db], numerical, atol=1e-8)

    def test_learns_labels_probabilities_and_regularization(self):
        base = LogisticRegressionClassifier(penalty='l2', reg_strength=.01).fit(self.X, self.y)
        strong = LogisticRegressionClassifier(penalty='l2', reg_strength=1).fit(self.X, self.y)
        np.testing.assert_array_equal(base.predict(self.X), self.y)
        np.testing.assert_allclose(base.predict_proba(self.X).sum(axis=1), 1)
        self.assertLess(np.linalg.norm(strong.coef_), np.linalg.norm(base.coef_))
        self.assertLess(base.loss_history_[-1], base.loss_history_[0])
        labels = np.where(self.y, 'canceled', 'booked')
        string_model = LogisticRegressionClassifier(penalty='l2').fit(self.X, labels)
        np.testing.assert_array_equal(string_model.predict(self.X), labels)

    def test_stability_at_extreme_logits(self):
        model = LogisticRegressionClassifier()
        X = np.array([[-1000.], [1000.]])
        loss, dw, db = model._loss_gradient(X, np.array([0., 1.]), np.ones(1), 0.)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(dw).all())
        self.assertTrue(np.isfinite(db))
        np.testing.assert_allclose(model._sigmoid(np.array([-1000., 1000.])), [0., 1.])

    def test_stochastic_reproducibility_and_persistence(self):
        kwargs = dict(penalty='l2', batch_size=1, max_iter=30, record_history=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            first = LogisticRegressionClassifier(**kwargs).fit(self.X, self.y)
            second = LogisticRegressionClassifier(**kwargs).fit(self.X, self.y)
        np.testing.assert_array_equal(first.coef_, second.coef_)
        self.assertEqual(len(first.parameter_history_), first.n_iter_ + 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pkl'
            first.save(path)
            restored = LogisticRegressionClassifier.load(path)
            np.testing.assert_array_equal(first.predict_proba(self.X), restored.predict_proba(self.X))
            path.write_bytes(pickle.dumps('wrong type'))
            with self.assertRaises(TypeError):
                LogisticRegressionClassifier.load(path)

    def test_invalid_inputs_and_failed_refit(self):
        for kwargs in ({'learning_rate': 0}, {'max_iter': True}, {'reg_strength': -1},
                       {'penalty': 'unknown'}, {'l1_ratio': 2}, {'tol': np.nan},
                       {'batch_size': 0}, {'random_state': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LogisticRegressionClassifier(**kwargs).fit(self.X, self.y)
        for y in ([0] * 4, [0, 1, 2, 2], [[0], [0], [1], [1]], [0, 0, 1, np.nan]):
            with self.subTest(y=y), self.assertRaises(ValueError):
                LogisticRegressionClassifier().fit(self.X, y)
        model = LogisticRegressionClassifier(penalty='l2').fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.predict(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            model.predict(self.X, threshold=np.nan)
        with self.assertRaises(ValueError):
            model.fit(self.X * np.nan, self.y)
        with self.assertRaises(RuntimeError):
            model.predict(self.X)

    def test_iteration_limit_is_reported(self):
        with self.assertWarns(RuntimeWarning):
            model = LogisticRegressionClassifier(max_iter=1).fit(self.X, self.y)
        self.assertFalse(model.converged_)
        self.assertEqual(model.n_iter_, 1)

    def test_parameters_score_and_invalidation(self):
        model = LogisticRegressionClassifier(penalty='l2').fit(self.X, self.y)
        self.assertEqual(model.score(self.X, self.y), 1.0)
        self.assertEqual(model.fit_status_, 0)
        for y in (self.y[:, None], [0], 0):
            with self.subTest(y=y), self.assertRaises(ValueError):
                model.score(self.X, y)
        params = model.get_params()
        self.assertEqual(LogisticRegressionClassifier(**params).get_params(), params)
        with self.assertRaises(ValueError):
            model.set_params(unknown=1)
        self.assertIs(model.set_params(reg_strength=.2), model)
        self.assertEqual(model.reg_strength, .2)
        self.assertFalse(hasattr(model, 'coef_'))
        with self.assertRaises(RuntimeError):
            model.predict(self.X)

    def test_failed_optimization_clears_fitted_state(self):
        model = LogisticRegressionClassifier(penalty='l2').fit(self.X, self.y)
        with patch.object(model, '_loss_gradient', side_effect=FloatingPointError):
            with self.assertRaises(FloatingPointError):
                model.fit(self.X, self.y)
        for name in ('coef_', 'classes_', 'fit_status_', 'loss_history_'):
            self.assertFalse(hasattr(model, name))
        with self.assertRaises(RuntimeError):
            model.predict(self.X)


if __name__ == '__main__':
    unittest.main()
