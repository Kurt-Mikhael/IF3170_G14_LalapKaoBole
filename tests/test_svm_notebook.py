import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "src/notebooks/main.ipynb"
SVM_SECTION_START = "## Persiapan Data dan Baseline SVM"
SVM_SECTION_END = "## Logistic Regression"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _svm_section(notebook):
    # Bagian SVM dicari lewat judulnya agar tes tidak bergantung pada nomor sel
    cells = notebook["cells"]

    def find(prefix):
        hits = [
            i for i, cell in enumerate(cells)
            if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith(prefix)
        ]
        assert len(hits) == 1, prefix
        return hits[0]

    return cells[find(SVM_SECTION_START):find(SVM_SECTION_END)]


def _execute_prep():
    namespace = {"__file__": str(NOTEBOOK)}
    notebook = _notebook()
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for index in (3, 4, 6):
            exec("".join(notebook["cells"][index]["source"]), namespace)  # noqa: S102 - trusted repository cells
        prep_cells = [
            cell for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "svm_train_indices, svm_valid_indices =" in "".join(cell["source"])
        ]
        assert len(prep_cells) == 1
        exec("".join(prep_cells[0]["source"]), namespace)  # noqa: S102 - trusted repository cells
    finally:
        os.chdir(previous_cwd)
    return notebook, namespace


class SvmNotebookTests(unittest.TestCase):
    def test_notebook_matches_jupyter_schema(self):
        nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))

    def test_notebook_preparation_has_leakage_free_split_and_fit(self):
        notebook, ns = _execute_prep()
        self.assertGreaterEqual(len(notebook["cells"]), 33)
        self.assertLessEqual(len(ns["svm_X_train"]), 1000)
        self.assertGreater(len(ns["svm_X_valid"]), 1000)
        self.assertTrue(set(ns["svm_train_indices"]).isdisjoint(ns["svm_valid_indices"]))

        hashes = ns["pd"].util.hash_pandas_object
        x = ns["train_df"].drop(columns=["is_canceled", "id"])
        groups = hashes(x, index=False)
        self.assertTrue(set(groups.loc[ns["svm_train_indices"]]).isdisjoint(
            groups.loc[ns["svm_valid_indices"]]
        ))
        self.assertEqual(ns["svm_X_train"].shape[1], ns["svm_X_valid"].shape[1])
        self.assertTrue(np.isfinite(ns["svm_X_train"]).all())
        self.assertTrue(np.isfinite(ns["svm_X_valid"]).all())
        self.assertNotIn("id", ns["svm_preprocessor"].feature_names_in_)
        numeric = ns["svm_numeric_features"]
        train_raw = x.iloc[ns["svm_train_indices"]]
        numeric_pipeline = ns["svm_preprocessor"].named_transformers_["numeric"]
        np.testing.assert_allclose(
            numeric_pipeline.named_steps["imputer"].statistics_,
            train_raw[numeric].median().to_numpy(),
        )
        imputed = numeric_pipeline.named_steps["imputer"].transform(train_raw[numeric])
        np.testing.assert_allclose(
            numeric_pipeline.named_steps["scaler"].mean_, imputed.mean(axis=0)
        )

    def test_notebook_sections_cover_both_solvers_plots_and_persistence(self):
        notebook = _notebook()
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        joined = "\n".join(sources)
        for name in (
            "svm_kernel_configs", "svm_sklearn_models", "svm_sklearn_results",
            "svm_evaluate_model", "svm_custom_models", "svm_custom_results",
            "svm_pca", "svm_visual_models", "plot_svm_boundary",
            "svm_preprocessor", "SVM.load",
        ):
            self.assertIn(name, joined)
        self.assertIn("from models.svm import SVM", joined)
        self.assertIn("from models.svm.visualization import plot_svm_boundary", joined)
        self.assertIn("SVM_MAX_ITER = 50000", joined)
        self.assertIn("('qp', 'smo')", joined)
        self.assertNotIn("max_iter=-1", joined)
        self.assertIn("max_iter': -1", joined)
        self.assertIn("OUTPUT_DIR / 'svm_boundaries_qp.png'", joined)
        self.assertIn("OUTPUT_DIR / 'svm_boundaries_smo.png'", joined)
        self.assertIn("svm_precision_recall.pdf", joined)
        self.assertIn("average_precision_score", joined)
        self.assertIn("precision_recall_curve", joined)
        self.assertIn("show_regions=True", joined)
        self.assertIn("bounds=svm_plot_bounds", joined)
        self.assertIn("legend=False", joined)
        pr_markdown = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown" and "Kurva precision-recall" in "".join(cell["source"])]
        self.assertEqual(len(pr_markdown), 1)
        pr_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "svm_precision_recall.pdf" in "".join(cell["source"])]
        self.assertEqual(len(pr_cells), 1)
        pca_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "svm_pca = PCA" in "".join(cell["source"]))
        self.assertNotIn("svm_precision_recall.pdf", "".join(pca_cell["source"]))
        self.assertLess(notebook["cells"].index(pr_markdown[0]), notebook["cells"].index(pr_cells[0]))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{cell.get('id', '')}", "exec")

    def test_new_cells_have_empty_outputs_and_imports_stay_in_library(self):
        for cell in _svm_section(_notebook()):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom))
                                 for node in ast.walk(tree)))

    def test_bounded_workflow_shared_parameters_plots_and_round_trip(self):
        notebook, ns = _execute_prep()
        ns["svm_X_train"] = ns["svm_X_train"][:40]
        ns["svm_y_train"] = ns["svm_y_train"].iloc[:40]
        ns["svm_X_valid"] = ns["svm_X_valid"][:50]
        ns["svm_y_valid"] = ns["svm_y_valid"].iloc[:50]
        ns["svm_train_indices"] = ns["svm_train_indices"][:40]
        ns["svm_valid_indices"] = ns["svm_valid_indices"][:50]
        with tempfile.TemporaryDirectory() as directory:
            ns["OUTPUT_DIR"] = Path(directory)
            for cell in _svm_section(notebook):
                source = "".join(cell["source"])
                # Sel persiapan sudah dijalankan oleh _execute_prep dan datanya sudah diperkecil di atas
                if cell["cell_type"] == "code" and "svm_train_indices, svm_valid_indices =" not in source:
                    exec(source, ns)  # noqa: S102 - trusted repository cells
            self.assertEqual(len(ns["svm_results"]), 9)
            self.assertEqual(len(ns["svm_custom_models"]), 6)
            self.assertEqual(len(ns["svm_visual_models"]), 6)
            for (solver, kernel), model in ns["svm_custom_models"].items():
                baseline = ns["svm_sklearn_models"][kernel]
                for param in ("C", "degree", "gamma", "coef0", "tol"):
                    self.assertEqual(model.get_params()[param], baseline.get_params()[param])
                self.assertEqual(model.max_iter, 50000)
                self.assertEqual(baseline.max_iter, -1)
                self.assertEqual(model.solver, solver)
                self.assertEqual(model.n_features_in_, ns["svm_X_train"].shape[1])
                self.assertEqual(ns["svm_visual_models"][(solver, kernel)].gamma, 0.5)
            np.testing.assert_allclose(ns["svm_pca"].mean_, ns["svm_X_train"].mean(axis=0))
            np.testing.assert_allclose(ns["svm_reloaded_X"], ns["svm_demo_X"])
            np.testing.assert_array_equal(ns["svm_reloaded_predictions"], ns["svm_demo_predictions"])
            for solver in ("qp", "smo"):
                path = Path(directory) / f"svm_boundaries_{solver}.png"
                self.assertGreater(path.stat().st_size, 1000)
                axes = ns["svm_boundary_axes"][solver]
                self.assertEqual(len(axes), 3)
                self.assertEqual(len(axes[0].figure.legends), 1)
                for ax in axes:
                    self.assertEqual(ax.get_xlim(), axes[0].get_xlim())
                    self.assertEqual(ax.get_ylim(), axes[0].get_ylim())
                    self.assertIsNone(ax.get_legend())
                    self.assertTrue(any("iter=" in line for line in ax.get_title().splitlines()))
                legend = {text.get_text() for text in axes[0].figure.legends[0].get_texts()}
                self.assertTrue({
                    "Observed 0 (not canceled)", "Observed 1 (canceled)",
                    "Predicted 0 (not canceled)", "Predicted 1 (canceled)",
                }.issubset(legend))
                self.assertEqual(axes[0].get_ylabel().startswith("PC2"), True)
                self.assertEqual([ax.get_ylabel() for ax in axes[1:]], ["", ""])
            pr_path = Path(directory) / "svm_precision_recall.pdf"
            self.assertGreater(pr_path.stat().st_size, 1000)
            self.assertEqual(set(ns["svm_pr_average_precision"]), {"linear", "poly", "rbf"})
            self.assertEqual(len(ns["svm_pr_ax"].lines), 4)
            self.assertEqual(len(ns["svm_pr_ax"].collections), 3)
            legend_text = [text.get_text() for text in ns["svm_pr_ax"].get_legend().get_texts()]
            self.assertTrue(any("proporsi positif" in text.lower() for text in legend_text))
            self.assertEqual(sum("AP=" in text for text in legend_text), 3)
            self.assertEqual(len(legend_text), 5)
            marker_paths = [
                tuple(np.round(collection.get_paths()[0].vertices, 4).ravel())
                for collection in ns["svm_pr_ax"].collections
            ]
            self.assertEqual(len(set(marker_paths)), 3)
            self.assertEqual(ns["svm_positive_prevalence"], np.mean(ns["svm_y_valid"] == 1))
            for index, svm_name in enumerate(("linear", "poly", "rbf")):
                model = ns["svm_custom_models"][("smo", svm_name)]
                scores = model.decision_function(ns["svm_X_valid"])
                np.testing.assert_array_equal(model.predict(ns["svm_X_valid"]), scores >= 0)
                np.testing.assert_allclose(
                    ns["svm_pr_ax"].collections[index].get_offsets(),
                    [[
                        ns["recall_score"](ns["svm_y_valid"], scores >= 0, pos_label=1, zero_division=0),
                        ns["precision_score"](ns["svm_y_valid"], scores >= 0, pos_label=1, zero_division=0),
                    ]],
                )
            self.assertEqual(ns["svm_pr_ax"].get_xlabel(), "Recall (kelas 1)")
            self.assertEqual(ns["svm_pr_ax"].get_ylabel(), "Precision (kelas 1)")
            for row in ns["svm_custom_results"] + ns["svm_sklearn_results"]:
                self.assertTrue({"backend", "solver", "kernel", "accuracy", "precision",
                                 "recall", "f1", "roc_auc", "train_time", "n_support",
                                 "n_iter", "converged", "fit_status"}.issubset(row))
                self.assertTrue(np.isfinite(row["roc_auc"]))
            ns["plt"].close("all")


if __name__ == "__main__":
    unittest.main()
