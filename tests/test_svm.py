import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import SVM
from models.svm import gaussian_kernel, linear_kernel, polynomial_kernel


class SVMTests(unittest.TestCase):
    def test_score_rejects_broadcasting_labels(self):
        model = SVM().fit(self.X, self.y)
        for y in (self.y[:, None], [0], 0):
            with self.subTest(y=y), self.assertRaises(ValueError):
                model.score(self.X, y)

    def test_failed_optimization_never_publishes_fitted_state(self):
        model = SVM(random_state=-1)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y)
        with self.assertRaises(RuntimeError):
            model.predict(self.X)
        model = SVM(solver='qp')
        with patch.object(model, '_fit_qp', side_effect=FloatingPointError), self.assertRaises(FloatingPointError):
            model.fit(self.X, self.y)
        with self.assertRaises(RuntimeError):
            model.predict(self.X)

    def test_small_C_support_reconstructs_scores(self):
        for solver in ('smo', 'qp'):
            for kernel in ('linear', 'rbf'):
                with self.subTest(solver=solver, kernel=kernel):
                    model = SVM(C=1e-8, solver=solver, kernel=kernel).fit(self.X, self.y)
                    self.assertGreater(len(model.support_), 0)
                    K = linear_kernel(self.X, model.support_vectors_)
                    if kernel == 'rbf':
                        K = gaussian_kernel(self.X, model.support_vectors_, gamma=model._gamma)
                    np.testing.assert_allclose(K @ model.dual_coef_[0] + model.intercept_[0], model.decision_function(self.X), atol=1e-20)
                    self.assertTrue(model.converged_)

    def test_single_pass_limit_checks_kkt_before_reporting_failure(self):
        with warnings.catch_warnings(record=True) as caught:
            model = SVM(max_passes=1).fit(self.X, self.y)
        self.assertTrue(model.converged_)
        self.assertEqual(model.fit_status_, 0)
        self.assertFalse(caught)

    def test_numeric_parameters_and_labels_are_validated(self):
        for kwargs in ({'gamma': []}, {'gamma': True}, {'gamma': 1j}, {'coef0': -1}, {'coef0': 'x'}, {'C': 'x'}, {'tol': 1j}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SVM(**kwargs).fit(self.X, self.y)
        for y in ([0, 0, np.nan, np.nan], [0, 0, np.inf, np.inf], [0j, 0j, 1j, 1j], [None, None, 'a', 'a']):
            with self.subTest(y=y), self.assertRaises(ValueError):
                SVM().fit(self.X, y)

    def test_smo_scan_switches_full_to_nonbound(self):
        # Observe local scan state without adding tracing hooks to the estimator.
        scans = []
        def trace(frame, event, arg):
            if frame.f_code.co_name == '_fit_smo' and event == 'line':
                state = frame.f_locals
                iteration = state.get('iteration')
                if iteration is not None and (not scans or scans[-1][0] != iteration):
                    scans.append((iteration, state['examine_all']))
            return trace
        sys.settrace(trace)
        try:
            SVM().fit(self.X, self.y)
        finally:
            sys.settrace(None)
        self.assertEqual(scans[:2], [(1, True), (2, False)])

    def test_nonlinear_and_nonseparable_match_sklearn(self):
        from sklearn.datasets import make_moons
        from sklearn.svm import SVC
        xor = (np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), np.array([0, 1, 1, 0]))
        for seed in (2, 13, 42):
            for X, y in (xor, make_moons(n_samples=50, noise=.23, random_state=seed)):
                for solver in ('smo', 'qp'):
                    for kernel in ('linear', 'poly', 'rbf'):
                        with self.subTest(seed=seed, solver=solver, kernel=kernel, n=len(y)):
                            model = SVM(kernel=kernel, solver=solver, random_state=seed, gamma=.8, coef0=1, tol=1e-4, max_iter=20000).fit(X, y)
                            ref = SVC(kernel=kernel, gamma=.8, coef0=1, tol=1e-7).fit(X, y)
                            self.assertTrue(model.converged_)
                            np.testing.assert_allclose(model.decision_function(X), ref.decision_function(X), atol=.015)
                            a = model._alphas
                            margin = (2 * y - 1) * model.decision_function(X)
                            self.assertLess(abs(a @ (2 * y - 1)), 1e-7)
                            self.assertTrue(np.all(margin[a < model.C - 1e-7] >= 1 - .002))
                            self.assertTrue(np.all(margin[a > 1e-7] <= 1 + .002))

    def test_seed_reproducibility_and_iteration_limit_status(self):
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=50, noise=.2, random_state=3)
        for solver in ('smo', 'qp'):
            with self.subTest(solver=solver):
                first = SVM(solver=solver, kernel='rbf', random_state=7).fit(X, y)
                second = SVM(solver=solver, kernel='rbf', random_state=7).fit(X, y)
                np.testing.assert_array_equal(first._alphas, second._alphas)
                self.assertEqual(first.n_iter_, second.n_iter_)
                with self.assertWarns(RuntimeWarning):
                    limited = SVM(solver=solver, kernel='rbf', max_iter=1).fit(X, y)
                self.assertFalse(limited.converged_)
                self.assertEqual(limited.fit_status_, 1)
                self.assertEqual(limited.n_iter_, 1)

    def setUp(self):
        self.X = np.array([[-2., -1.], [-1., -1.], [1., 1.], [2., 1.]])
        self.y = np.array([0, 0, 1, 1])

    def test_kernels(self):
        X = np.array([[1., 2.], [2., 0.]])
        Z = np.array([[1., 1.]])
        np.testing.assert_allclose(linear_kernel(X, Z), [[3], [2]])
        np.testing.assert_allclose(polynomial_kernel(X, Z, degree=2, gamma=.5, coef0=1), [[6.25], [4.]])
        np.testing.assert_allclose(gaussian_kernel(X, Z, gamma=1), np.exp([[-1], [-2]]))

    def test_solver_kernel_matrix_fits_feasibly_and_recovers_labels(self):
        for solver in ('smo', 'qp'):
            for kernel in ('linear', 'polynomial', 'gaussian'):
                with self.subTest(solver=solver, kernel=kernel):
                    model = SVM(solver=solver, kernel=kernel, max_iter=5000, tol=1e-4).fit(self.X, self.y)
                    self.assertIsNotNone(model.fit(self.X, self.y))
                    np.testing.assert_array_equal(model.predict(self.X), self.y)
                    self.assertGreaterEqual(model.score(self.X, self.y), .99)
                    self.assertAlmostEqual(np.dot(model._alphas, np.where(self.y == model.classes_[1], 1., -1.)), 0, places=6)
                    self.assertTrue(np.all(model._alphas >= -1e-8))
                    self.assertTrue(np.all(model._alphas <= model.C + 1e-8))
                    self.assertEqual(model.decision_function(self.X).shape, (4,))
                    self.assertEqual(model.n_iter_ >= 0, True)

    def test_labels_validation_refit_and_parameter_invalidation(self):
        model = SVM().fit(self.X, self.y)
        with self.assertRaises(ValueError): model.predict([[1.]])
        with self.assertRaises(ValueError): model.fit(self.X, [0, 0, 0, 0])
        with self.assertRaises(ValueError): model.fit(self.X, [0, 1, 2, 2])
        with self.assertRaises(ValueError): model.fit(self.X[:2], self.y)
        with self.assertRaises(ValueError): model.fit([[0, np.nan], [1, 2]], [0, 1])
        with self.assertRaises(RuntimeError): model.predict(self.X)
        model.set_params(C=2)
        with self.assertRaises(RuntimeError): model.predict(self.X)
        self.assertEqual(model.get_params()['C'], 2)

    def test_gamma_modes_and_aliases(self):
        for gamma in ('scale', 'auto', .25):
            for kernel in ('poly', 'rbf'):
                model = SVM(kernel=kernel, gamma=gamma, solver='smo').fit(self.X, self.y)
                self.assertTrue(np.isfinite(model.decision_function(self.X)).all())

    def test_duplicate_conflicting_points_are_finite_and_predict(self):
        X = np.array([[-1.], [-1.], [1.], [1.], [0.], [0.]])
        y = np.array([0, 0, 1, 1, 0, 1])
        for solver in ('smo', 'qp'):
            model = SVM(solver=solver, max_iter=5000).fit(X, y)
            self.assertTrue(np.isfinite(model.decision_function(X)).all())
            self.assertTrue(np.isfinite(model.intercept_).all())
            self.assertTrue(np.isfinite(model._alphas).all())

    def test_seeded_fixture_satisfies_dual_kkt_conditions(self):
        rng = np.random.default_rng(23)
        X = np.r_[rng.normal((-1, -1), .25, (12, 2)), rng.normal((1, 1), .25, (12, 2))]
        y = np.r_[np.zeros(12), np.ones(12)]
        for solver in ('smo', 'qp'):
            model = SVM(solver=solver, kernel='linear', C=2, tol=1e-4, max_iter=10000).fit(X, y)
            margin = np.where(y == model.classes_[1], 1., -1.) * model.decision_function(X)
            alpha = model._alphas
            free = (alpha > 1e-6) & (alpha < model.C - 1e-6)
            self.assertTrue(np.all(margin[alpha <= 1e-6] >= 1 - 2e-3))
            self.assertTrue(np.all(margin[alpha >= model.C - 1e-6] <= 1 + 2e-3))
            self.assertTrue(np.all(np.abs(margin[free] - 1) <= 2e-3))

    def test_params_serialization_and_exports(self):
        model = SVM().fit(self.X, self.y)
        params = model.get_params()
        self.assertEqual(params['solver'], 'smo')
        model.set_params(solver='qp')
        self.assertEqual(model.solver, 'qp')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'model.pkl'
            # re-fit after parameter change invalidated the old fitted state
            model.fit(self.X, self.y).save(path)
            loaded = SVM.load(path)
            np.testing.assert_array_equal(loaded.predict(self.X), self.y)
        self.assertTrue(model.converged_)
        self.assertIn(model.fit_status_, (0, 1))
        self.assertEqual(model.support_vectors_.shape[1], 2)
        self.assertEqual(model.dual_coef_.shape, (1, len(model.support_)))
        self.assertEqual(model.coef_.shape, (1, 2))

    def test_invalid_parameters(self):
        for kwargs in ({'C': 0}, {'tol': 0}, {'max_iter': 0}, {'solver': 'other'}, {'gamma': -1}, {'kernel': 'bad'}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                SVM(**kwargs).fit(self.X, self.y)

    def test_matches_sklearn_on_separable_fixture(self):
        from sklearn.svm import SVC
        for kernel in ('linear', 'poly', 'rbf'):
            custom = SVM(kernel={'poly': 'polynomial', 'rbf': 'gaussian'}.get(kernel, kernel), solver='smo', gamma='scale', max_iter=5000).fit(self.X, self.y)
            reference = SVC(kernel=kernel, gamma='scale').fit(self.X, self.y)
            np.testing.assert_array_equal(custom.predict(self.X), reference.predict(self.X))


if __name__ == '__main__':
    unittest.main()
