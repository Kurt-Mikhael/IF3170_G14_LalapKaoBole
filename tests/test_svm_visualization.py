import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.svm.svm import SVM
from models.svm.visualization import plot_svm_boundary


class AnalyticModel:
    classes_ = np.array(["left", "right"])
    n_features_in_ = 2
    support_vectors_ = np.array([[-1.0, 0.0], [1.0, 0.0]])

    def decision_function(self, X):
        return np.asarray(X)[:, 0]


class ConstantModel(AnalyticModel):
    def decision_function(self, X):
        return np.ones(len(X))


class SVMVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 1.0]])
        self.y = np.array(["left", "left", "right", "right"])

    def tearDown(self):
        plt.close("all")

    def test_draws_three_decision_levels_and_documents_support_vectors(self):
        ax = plot_svm_boundary(
            AnalyticModel(), self.X, self.y, title="Analytic boundary"
        )
        self.assertEqual(ax.get_title(), "Analytic boundary")
        self.assertEqual(
            len(ax.collections), 6
        )  # Three contours, two classes, and support markers.
        self.assertEqual(len(ax.lines), 0)
        self.assertTrue(
            any(
                len(collection.get_offsets()) == 2
                and len(collection.get_facecolors()) == 0
                for collection in ax.collections
            )
        )
        legend = {text.get_text() for text in ax.get_legend().get_texts()}
        self.assertEqual(
            legend, {"left", "right", "Support vectors", "f = -1", "f = 0", "f = +1"}
        )
        legend_lines = {
            text.get_text(): handle.get_linestyle()
            for text, handle in zip(
                ax.get_legend().get_texts(), ax.get_legend().legend_handles
            )
            if hasattr(handle, "get_linestyle")
        }
        self.assertEqual(
            [legend_lines[level] for level in ("f = -1", "f = 0", "f = +1")],
            ["--", "-", "--"],
        )
        self.assertEqual(ax.get_xlabel(), "Feature 1")
        self.assertEqual(ax.get_ylabel(), "Feature 2")

    def test_reuses_supplied_axes_and_leaves_figure_ownership_to_caller(self):
        fig, ax = plt.subplots()
        self.assertIs(plot_svm_boundary(AnalyticModel(), self.X, self.y, ax=ax), ax)
        self.assertIs(ax.figure, fig)

    def test_constant_scores_annotate_missing_contours_without_inventing_them(self):
        ax = plot_svm_boundary(ConstantModel(), self.X, self.y)
        self.assertTrue(
            any("constant decision score" in text.get_text() for text in ax.texts)
        )
        self.assertEqual(len(ax.collections), 3)

    def test_empty_endpoint_contour_is_not_advertised(self):
        class EndpointModel(AnalyticModel):
            def decision_function(self, X):
                return np.asarray(X)[:, 0] / 2.2

        ax = plot_svm_boundary(EndpointModel(), self.X, self.y)
        legend = {text.get_text() for text in ax.get_legend().get_texts()}
        self.assertNotIn("f = +1", legend)
        self.assertTrue(any("f = +1" in text.get_text() for text in ax.texts))

    def test_real_svm_kernels_and_solvers_render_actual_contours_and_support_vectors(
        self,
    ):
        rng = np.random.default_rng(31)
        X = np.vstack(
            (rng.normal((-2, -2), 0.35, (12, 2)), rng.normal((2, 2), 0.35, (12, 2)))
        )
        y = np.array([0] * 12 + [1] * 12)
        with tempfile.TemporaryDirectory() as directory:
            for solver in ("smo", "qp"):
                for kernel in ("linear", "polynomial", "gaussian"):
                    with self.subTest(solver=solver, kernel=kernel):
                        model = SVM(
                            solver=solver, kernel=kernel, random_state=31, gamma=0.5
                        ).fit(X, y)
                        ax = plot_svm_boundary(model, X, y, resolution=100)
                        contour_collections = ax.collections[3:]
                        self.assertEqual(len(contour_collections), 3)
                        for collection, level in zip(contour_collections, (-1, 0, 1)):
                            vertices = np.concatenate(
                                [path.vertices for path in collection.get_paths()]
                            )
                            self.assertGreater(len(vertices), 0)
                            np.testing.assert_allclose(
                                model.decision_function(vertices), level, atol=0.08
                            )
                        support_artist = next(
                            collection
                            for collection in ax.collections[:3]
                            if len(collection.get_facecolors()) == 0
                        )
                        np.testing.assert_array_equal(
                            support_artist.get_offsets(), model.support_vectors_
                        )
                        output = Path(directory) / f"{solver}-{kernel}.png"
                        ax.figure.savefig(output)
                        self.assertGreater(output.stat().st_size, 0)
                        plt.close(ax.figure)

    def test_batched_grid_evaluation_includes_support_points(self):
        class RecordingModel(AnalyticModel):
            def __init__(self):
                self.batch_sizes = []

            def decision_function(self, X):
                self.batch_sizes.append(len(X))
                return super().decision_function(X)

        model = RecordingModel()
        plot_svm_boundary(model, self.X, self.y, resolution=2)
        self.assertTrue(all(size <= 512 for size in model.batch_sizes))
        self.assertTrue(all(size > 0 for size in model.batch_sizes))

    def test_saves_a_nonempty_png(self):
        ax = plot_svm_boundary(AnalyticModel(), self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.png"
            ax.figure.savefig(path)
            self.assertGreater(path.stat().st_size, 0)

    def test_rejects_invalid_inputs_and_unfitted_models(self):
        for X, y in (
            ([], []),
            ([[0, np.nan]], ["left"]),
            ([[0, 1]], []),
            ([[0]], ["left"]),
        ):
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                plot_svm_boundary(AnalyticModel(), X, y)
        for resolution in (True, 1, 2.5, "3"):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                plot_svm_boundary(
                    AnalyticModel(), self.X, self.y, resolution=resolution
                )
        with self.assertRaises(ValueError):
            plot_svm_boundary(
                AnalyticModel(), self.X, np.array(["left", "left", "right", "other"])
            )

        class Unfitted:
            pass

        with self.assertRaisesRegex(ValueError, "fitted"):
            plot_svm_boundary(Unfitted(), self.X, self.y)

        class InvalidFeatures(AnalyticModel):
            n_features_in_ = 3

        with self.assertRaisesRegex(ValueError, "two features"):
            plot_svm_boundary(InvalidFeatures(), self.X, self.y)

        class NonfiniteScores(AnalyticModel):
            def decision_function(self, X):
                return np.full(len(X), np.nan)

        with self.assertRaisesRegex(ValueError, "finite score"):
            plot_svm_boundary(NonfiniteScores(), self.X, self.y)


if __name__ == "__main__":
    unittest.main()
