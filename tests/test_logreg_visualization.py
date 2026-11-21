import sys
import unittest
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import LogisticRegressionClassifier
from models.logreg.visualization import _log_loss_at, create_logreg_training_animation


class LogisticRegressionVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.], [-1.], [0.], [0.], [1.], [2.]])
        self.y = np.array([0, 0, 1, 0, 1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.model = LogisticRegressionClassifier(max_iter=15, record_history=True).fit(self.X, self.y)

    def tearDown(self):
        plt.close("all")

    def test_surface_matches_model_loss_and_is_stable_at_extreme_scores(self):
        parameters = np.array([[0., 0.], [.3, -.7], [-1000., 1000.]])
        actual = _log_loss_at(parameters, self.X[:, 0], self.y, np.ones(len(self.y)))
        expected = [-self.model._log_likelihood_gradient(self.X, self.y, np.array([w]), b)[0]
                    for b, w in parameters]
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.isfinite(actual).all())

    def test_animation_uses_actual_bias_weight_history_and_losses(self):
        video = create_logreg_training_animation(self.model, self.X, self.y, grid_size=12)
        video._fig.canvas.draw()
        artists = video._func(self.model.n_iter_)
        expected = np.array([[b, w[0]] for w, b in self.model.parameter_history_])
        np.testing.assert_array_equal(artists[0].get_xdata(), expected[:, 0])
        np.testing.assert_array_equal(artists[0].get_ydata(), expected[:, 1])
        np.testing.assert_allclose(artists[2].get_ydata(), self.model.loss_history_)
        self.assertIn(f"Epoch {self.model.n_iter_}/", artists[-1].get_text())

    def test_rejects_a_surface_that_does_not_describe_the_recorded_training(self):
        for X, y in ((self.X + 2, self.y), (self.X, 1 - self.y)):
            with self.subTest(X=X, y=y), self.assertRaisesRegex(ValueError, "same X/y"):
                create_logreg_training_animation(self.model, X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            regularized = LogisticRegressionClassifier(penalty="l2", max_iter=1, record_history=True).fit(self.X, self.y)
            no_history = LogisticRegressionClassifier(max_iter=1).fit(self.X, self.y)
            multi = LogisticRegressionClassifier(max_iter=1, record_history=True).fit(np.c_[self.X, self.X], self.y)
        for model, X, message in ((regularized, self.X, "penalty=None"),
                                   (no_history, self.X, "record_history=True"),
                                   (multi, np.c_[self.X, self.X], "one-feature")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                create_logreg_training_animation(model, X, self.y)


if __name__ == "__main__":
    unittest.main()
