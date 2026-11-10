import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "src/notebooks/v1.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text())


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
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{cell.get('id', '')}", "exec")

    def test_new_cells_have_empty_outputs_and_imports_stay_in_library(self):
        for index, cell in enumerate(_notebook()["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            if index >= 33:
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
            for cell in notebook["cells"][36:]:
                if cell["cell_type"] == "code":
                    exec("".join(cell["source"]), ns)  # noqa: S102 - trusted repository cells
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
            for row in ns["svm_custom_results"] + ns["svm_sklearn_results"]:
                self.assertTrue({"backend", "solver", "kernel", "accuracy", "precision",
                                 "recall", "f1", "roc_auc", "train_time", "n_support",
                                 "n_iter", "converged", "fit_status"}.issubset(row))
                self.assertTrue(np.isfinite(row["roc_auc"]))
            ns["plt"].close("all")


if __name__ == "__main__":
    unittest.main()
