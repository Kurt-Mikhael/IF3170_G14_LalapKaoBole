import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "src/notebooks/main.ipynb"
sys.path.insert(0, str(ROOT / "src"))
from models import LogisticRegressionClassifier
from models.svm import SVM


class SubmissionModelExportTests(unittest.TestCase):
    def test_final_cell_exports_selected_models_used_for_submissions(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cv_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "svm_custom_table =" in "".join(cell["source"])
        )
        selection_start = cv_source.index("svm_custom_table =")
        selection_end = cv_source.index("\n\n\n# Setiap fungsi", selection_start)
        selection_source = cv_source[selection_start:selection_end]
        final_cell = notebook["cells"][-1]
        self.assertEqual(final_cell["cell_type"], "code")
        source = "".join(final_cell["source"])

        train = pd.DataFrame({"feature": [-3., -2., -1., 1., 2., 3.]})
        labels = np.array([0, 0, 0, 1, 1, 1])
        svm_preprocessor = StandardScaler().fit(train)
        svm_selected = SVM(kernel="linear", C=1, max_iter=100, solver="smo").fit(
            svm_preprocessor.transform(train), labels
        )
        svm_demo = SVM(kernel="rbf", C=1, max_iter=100, solver="smo").fit(
            svm_preprocessor.transform(train), 1 - labels
        )
        logreg_preprocessor = StandardScaler().fit(train)
        logreg_selected = LogisticRegressionClassifier(penalty="l2", reg_strength=0.01, max_iter=10000).fit(
            logreg_preprocessor.transform(train), labels
        )
        logreg_demo = LogisticRegressionClassifier(penalty="l2", reg_strength=1.0, max_iter=10000).fit(
            logreg_preprocessor.transform(train), 1 - labels
        )

        class OtherModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            raw_test = pd.DataFrame({"feature": [-2.5, -0.2, 0.3, 2.5]})
            pd.DataFrame({"id": [1, 2, 3, 4]}).to_csv(output / "sample_submission.csv", index=False)
            svm_custom_models = {("smo", "linear"): svm_selected, ("smo", "rbf"): svm_demo}
            logreg_custom_models = {"selected": logreg_selected, "l2": logreg_demo}
            ns = {
                "OUTPUT_DIR": output,
                "DATA_DIR": output,
                "pd": pd,
                "np": np,
                "pickle": pickle,
                "Path": Path,
                "SVM": SVM,
                "test_df": raw_test.assign(id=[1, 2, 3, 4]),
                "train_df": pd.DataFrame({"feature": train["feature"], "id": range(6), "is_canceled": labels}),
                "logreg_X_raw": train.copy(),
                "SEED": 17,
                "LOGREG_STRENGTH": 0.001,
                "logreg_train_idx": np.array([0, 1, 2]),
                "logreg_valid_idx": np.array([3, 4, 5]),
                "sklearn_version": "test-version",
                "svm_results": pd.DataFrame([
                    {"backend": "custom", "solver": "smo", "kernel": "rbf", "f1": 0.25},
                    {"backend": "custom", "solver": "smo", "kernel": "linear", "f1": 0.9},
                    {"backend": "sklearn", "solver": "sklearn", "kernel": "linear", "f1": 1.0},
                ]),
                "svm_custom_models": svm_custom_models,
                "svm_preprocessor": svm_preprocessor,
                "logreg_results": pd.DataFrame([
                    {"implementation": "custom", "variant": "l2", "f1": 0.25},
                    {"implementation": "custom", "variant": "selected", "f1": 0.9},
                    {"implementation": "scikit-learn", "variant": "external", "f1": 1.0},
                ]),
                "logreg_custom_models": logreg_custom_models,
                "logreg_preprocessor": logreg_preprocessor,
                "dtl_final_model": OtherModel(),
                "dtl_preprocess": lambda raw, rare: raw.to_numpy(),
                "dtl_final_rare": None,
                "knn_final_model": OtherModel(),
                "knn_final_preprocessor": StandardScaler().fit(train),
                "nb_final_model": OtherModel(),
                "nb_transform": lambda raw, params, mode: raw.to_numpy(),
                "nb_final_params": None,
                "NB_MODE": None,
            }
            exec(selection_source, ns)  # noqa: S102 - trusted notebook selection assignments only
            self.assertIs(ns["svm_best_model"], svm_selected)
            self.assertIs(ns["logreg_best_model"], logreg_selected)
            exec(source, ns)  # noqa: S102 - execute trusted final notebook cell

            svm_csv = pd.read_csv(output / "submission_svm.csv")["is_canceled"].to_numpy()
            with (output / "svm_preprocessor.pkl").open("rb") as file:
                saved_svm_preprocessor = pickle.load(file)
            saved_svm_model = SVM.load(output / "svm.pkl")
            svm_saved_predictions = saved_svm_model.predict(
                saved_svm_preprocessor.transform(raw_test)
            )
            np.testing.assert_array_equal(svm_saved_predictions, svm_csv)
            np.testing.assert_array_equal(svm_csv, [0, 0, 1, 1])
            self.assertFalse(np.array_equal(svm_csv, svm_demo.predict(svm_preprocessor.transform(raw_test))))

            logreg_csv = pd.read_csv(output / "submission_logreg.csv")["is_canceled"].to_numpy()
            with (output / "logreg_bundle.pkl").open("rb") as file:
                bundle = pickle.load(file)
            logreg_saved_predictions = bundle["model"].predict(
                bundle["preprocessor"].transform(raw_test[bundle["feature_columns"]])
            )
            np.testing.assert_array_equal(logreg_saved_predictions, logreg_csv)
            np.testing.assert_array_equal(logreg_csv, [0, 0, 1, 1])
            self.assertFalse(np.array_equal(logreg_csv, logreg_demo.predict(logreg_preprocessor.transform(raw_test))))
            expected_keys = {
                "model", "preprocessor", "feature_columns", "seed", "reg_strength",
                "train_indices", "validation_indices", "sklearn_version",
            }
            self.assertEqual(expected_keys - bundle.keys(), set())


if __name__ == "__main__":
    unittest.main()
