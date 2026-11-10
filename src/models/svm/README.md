# Support Vector Machine

Support vector machines (SVMs) classify data by finding a decision boundary with a wide margin between two classes. This implementation supports linear, polynomial, and Gaussian kernels, with two training methods: quadratic programming (QP) and sequential minimal optimization (SMO). Model calculations use NumPy.

Input features must be numeric and finite. Missing-value handling, categorical encoding, and feature scaling are performed before fitting. Preprocessing is fitted on the training data and applied unchanged to validation or test data to prevent information leakage.

## Decision boundary and soft margin

For an input vector $x$, the linear decision function is defined by a weight vector $w$ and an intercept $b$:

$$
f(x) = w^T x + b.
$$

The weights determine the boundary's orientation, while the intercept determines its position. The sign of the decision score selects the predicted class; the score is not a probability. Original labels are sorted and represented internally as $-1$ and $+1$. Negative scores select the first label, and nonnegative scores select the second.

The index $i$ identifies a training sample, with features $x_i$ and label $y_i$.

The decision boundary has a score of zero. For a nonzero $w$, the parallel surfaces at scores $-1$ and $+1$ have a separation of:

$$
\text{margin width} = \frac{2}{\lVert w \rVert}.
$$

Reducing the weight magnitude $\lVert w \rVert$ increases the margin width. A soft-margin formulation also allows margin violations, accommodating overlapping classes and misclassified training samples.

The linear training objective balances margin width against these violations:

$$
\begin{aligned}
\text{minimize}_{w,b,\xi}\quad & \frac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i \\
\text{subject to}\quad & y_i(w^T x_i+b) \geq 1-\xi_i, \\
& \xi_i \geq 0.
\end{aligned}
$$

The slack variable $\xi_i$ measures the margin violation for sample $i$. A larger $C$ penalizes violations more strongly; a smaller $C$ places more emphasis on a wider margin. The choice affects the model's fit but does not guarantee a particular accuracy.

## Kernel functions

Kernel functions measure similarity between samples and allow nonlinear decision boundaries without explicitly constructing additional features. The supported kernels are:

- Linear: uses the dot product and produces a straight boundary in two dimensions.
- Polynomial: represents combinations of features and can produce curved boundaries.
- Gaussian: assigns higher similarity to nearby samples and lower similarity to distant samples.

$$
\begin{aligned}
K_{\text{linear}}(x,z) &= x^Tz, \\
K_{\text{poly}}(x,z) &= (\gamma x^Tz + c_0)^d, \\
K_{\text{Gaussian}}(x,z) &= \exp(-\gamma\lVert x-z\rVert^2).
\end{aligned}
$$

In these expressions, $x$ and $z$ are two feature rows, $d$ is the polynomial degree, $c_0$ is the polynomial offset, and $\gamma$ controls the influence of dot products or distances. The code accepts `"polynomial"` or `"poly"` for the polynomial kernel, and `"gaussian"` or `"rbf"` for the Gaussian kernel.

For the polynomial and Gaussian kernels, `gamma="auto"` sets $\gamma$ to the reciprocal of the number of features. `gamma="scale"`, the default, uses the variance of all entries in the training feature array:

$$
\gamma_{\text{auto}} = \frac{1}{n_{\text{features}}},
\qquad
\gamma_{\text{scale}} = \frac{1}{n_{\text{features}}\,\operatorname{Var}(X)}.
$$

If that variance is zero, the implementation uses $\gamma=1$. A positive numeric value can also be supplied. Gamma has no effect on the linear kernel.

These kernels correspond to feature mappings $\phi(x)$ satisfying $K(x,z)=\phi(x)^T\phi(z)$. The implementation evaluates the kernel directly without constructing the mapped features.

## Support vectors and prediction

Training determines a coefficient $\alpha_i$ for each sample. Its contribution to the decision score is weighted by this coefficient and its class label:

$$
f(x) = \sum_i \alpha_i y_i K(x_i,x) + b.
$$

The coefficients satisfy a bound on each value and a balance condition across labels:

$$
0 \leq \alpha_i \leq C,
\qquad
\sum_i \alpha_i y_i = 0.
$$

Samples with positive coefficients are support vectors. At a solution, coefficients strictly between zero and $C$ correspond to samples on the margin. Samples with coefficients at $C$ may lie on the margin, within it, or on the wrong side of the decision boundary. The implementation includes every strictly positive coefficient in `support_`, including small positive values caused by numerical precision.

For the linear kernel, the model combines the support-vector contributions into a single weight vector:

$$
w = \sum_i \alpha_i y_i x_i,
\qquad
f(x)=w^Tx+b.
$$

Prediction for this case uses `coef_`. Polynomial and Gaussian predictions instead evaluate the kernel between each new row and the stored support vectors. The public `decision_function` returns one score per input row; `predict` maps its sign back to the original labels.

## Training algorithms

Both solvers minimize the same objective under the same coefficient constraints. The `"qp"` solver updates the full coefficient vector using projected gradient steps, while `"smo"` updates pairs of coefficients.

### Quadratic Programming (QP) with projected gradient

The QP solver minimizes a quadratic objective, then projects each update back onto the coefficient constraints. Momentum accelerates the updates.

For training labels $y_i$ and kernel values $K(x_i,x_j)$, the QP solver forms a matrix $Q$:

$$
Q_{ij}=y_i y_j K(x_i,x_j).
$$

The objective $J$ and its gradient are:

$$
J(\alpha)=\frac{1}{2}\alpha^TQ\alpha-\sum_i\alpha_i,
\qquad
\nabla J(\alpha)=Q\alpha-\mathbf{1}.
$$

With $\mathbf{1}$ denoting a vector of ones, the gradient step from a trial vector $z$ uses step size $t$:

$$
v=z-t\left(Qz-\mathbf{1}\right).
$$

The step size is the reciprocal of the largest eigenvalue of $Q$, which bounds how quickly the gradient changes. A minimum value of $10^{-12}$ is applied before division to avoid a zero denominator.

The resulting vector $v$ is projected onto the coefficient bounds and label-balance constraint:

$$
\alpha_i = \operatorname{clip}(v_i-\lambda y_i,0,C),
\qquad
\sum_i \alpha_i y_i=0.
$$

The clipping operation enforces the individual bounds. The scalar $\lambda$ is determined by 80 bisection steps to satisfy the balance condition within numerical precision. Independent clipping alone would not enforce this condition. The projection is implemented in `_project`.

Momentum incorporates part of the previous update into the next trial. Since this trial can fall outside the constraints, each subsequent gradient step is projected again. A sufficiently small projected update triggers intercept recovery and a check of the training margins. A final margin check is also performed at the iteration limit. These operations are implemented in `_fit_qp`.

### Sequential Minimal Optimization (SMO)

SMO optimizes two coefficients at a time while preserving $\sum_i\alpha_i y_i=0$. The solver scans for samples that violate the margin conditions and maintains a cache of prediction errors:

$$
E_j=f(x_j)-y_j.
$$

For a sample requiring an update, the preferred partner has a coefficient strictly between zero and $C$ and the largest difference in cached error. Such coefficients are called non-bound. If the preferred pair cannot improve the objective, the solver searches the remaining non-bound samples and then all samples, using randomly selected starting positions. Bound comparisons allow for numerical tolerance.

For a selected pair $i,j$, the bounds $L$ and $H$ on the second coefficient depend on their labels:

$$
\begin{array}{ll}
y_i\ne y_j: & L=\max(0,\alpha_j-\alpha_i),\quad H=\min(C,C+\alpha_j-\alpha_i),\\
y_i=y_j: & L=\max(0,\alpha_i+\alpha_j-C),\quad H=\min(C,\alpha_i+\alpha_j).
\end{array}
$$

These bounds preserve both the coefficient limits and the label-weighted sum. The curvature of the objective along the pair's update direction is:

$$
\eta=K(x_i,x_i)+K(x_j,x_j)-2K(x_i,x_j).
$$

When $\eta>0$, the update is:

$$
\begin{aligned}
\alpha_j^{\text{new}} &= \operatorname{clip}\left(\alpha_j+\frac{y_j(E_i-E_j)}{\eta},L,H\right),\\
\alpha_i^{\text{new}} &= \alpha_i+y_i y_j(\alpha_j-\alpha_j^{\text{new}}).
\end{aligned}
$$

If $\eta\leq0$, the solver compares the objective at $L$ and $H$ and selects the better endpoint. Numerically indistinguishable endpoint values leave the pair unchanged. This case includes identical feature rows, which can produce zero curvature.

After each successful update, SMO computes intercept candidates $b_1$ and $b_2$ from the two updated samples. It selects the candidate associated with a coefficient strictly between zero and $C$, or averages the candidates when neither coefficient is in that range. The cached errors are then adjusted using the changed coefficients and intercept. These operations are implemented in `_fit_smo`.

The SMO loop alternates between scanning every row and scanning only non-bound rows. Its `n_iter_` count is the number of these outer scans, not the number of pair attempts. The QP solver counts projected steps, so the two iteration counts are not directly comparable.

## Intercept estimation and convergence

The signed margin $m_i=y_i f(x_i)$ is positive for samples on the correct side of the decision boundary and negative for samples on the wrong side. At a solution, the coefficients and margins satisfy:

$$
\begin{array}{ll}
\alpha_i=0 &: m_i\geq1,\\
0<\alpha_i<C &: m_i=1,\\
\alpha_i=C &: m_i\leq1.
\end{array}
$$

Both solvers check these conditions within the tolerance `tol`. A sample with a coefficient strictly between zero and $C$ is a free support vector and provides the following intercept estimate:

$$
b_i=y_i-\sum_j\alpha_jy_jK(x_j,x_i).
$$

QP averages the estimates from free support vectors. When none exist, it uses the midpoint of the intercept interval allowed by the margin conditions, provided both bounds are available. Otherwise, it falls back to the average difference between the labels and the scores without an intercept. SMO determines the intercept during its pair updates.

If training stops before these conditions are satisfied, the model sets `converged_` to `False` and `fit_status_` to `1`, and emits a runtime warning. The fitted model remains available, but its result is an approximation that has not met the stopping tolerance. The default iteration limit is 1,000 and may need to be increased for more demanding fits.

## Decision boundary visualization

The visualization displays contours at $f(x)=-1$, $f(x)=0$, and $f(x)=+1$ when present in the plotted region. The zero contour represents the decision boundary, while the other two represent margin levels. For nonlinear kernels, these levels describe decision scores rather than Euclidean distances. Support vectors are marked with circles.

Visualization requires a model fitted on two features. For higher-dimensional data, the notebook fits PCA on the training data and trains separate SVMs on the resulting two-dimensional coordinates. These plots show the boundaries of the two-dimensional models, not projections of the original models' boundaries. The polynomial QP visualization did not converge within its configured limit, and this status is reported.

## Computational requirements

Training builds a dense matrix of all pairwise kernel values. For $n$ training rows, that matrix has $n^2$ entries and uses about $8n^2$ bytes in float64, before temporary arrays. With 27,373 rows, the matrix alone requires about 6.0 GB. The notebook uses a stratified subset of 1,000 rows for its main kernel comparison; those results are not full-dataset results.

QP also computes a largest eigenvalue, which can require cubic work in the number of rows, and its steps use dense matrix-vector products. SMO updates pairs but still stores and accesses the same full kernel matrix. Neither method avoids the dense-memory cost. The implementation uses NumPy for model calculations; the plotting helper uses Matplotlib.
