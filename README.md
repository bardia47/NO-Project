# Robust Parameter Estimation via Smooth L1-Regularization

**Authors:** Bardia Dorry (5866102), Sebastian James (6000546)

## Idea

When fitting a regression line to empirical data, standard Ordinary Least Squares ($L_2$ loss) is highly sensitive to outliers because it penalizes residuals quadratically. In contrast, $L_1$ regression (minimizing absolute residuals) provides a robust alternative since the penalty scales linearly, preventing anomalous spikes from heavily distorting the model parameters.

To evaluate this behavior under real-world conditions, this project implements and compares three optimization objectives using historical atmospheric data from the [Beijing Air Pollution dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv):
1. **$L_2$ Regression (Ordinary Least Squares):** Closed-form singular value decomposition optimization.
2. **$L_1$ Regression (Robust Linear Program):** Formulated as a smooth constrained Linear Program (LP) by introducing slack variables $s$:
   $$\text{minimize } \sum_{i=1}^{m} s_i$$
   $$\text{subject to } |Ax - y| \leq s \quad \text{and} \quad s \geq 0$$
3. **Huber Regression (Hyper-Balanced):** A smooth, piecewise loss function optimized via the quasi-Newton BFGS algorithm, combining the benefits of $L_2$ stability for small residuals and $L_1$ robustness for severe outliers.

---

## Results

The models were evaluated using empirical data slices mapping **Cumulative Wind Speed ($Iws$)** against **$\text{PM2.5}$ Pollution Concentration**. The top 20% of the highest baseline $L_2$ residuals were automatically flagged as systemic weather anomalies or severe smog events.

### Model Parameter & Error Metrics
Below is the baseline comparison across 100 historical atmospheric data points:

| Optimization Method | Estimated Slope | Estimated Intercept | Robustness Behavior |
| :--- | :---: | :---: | :--- |
| **$L_2$ (Least Squares)** | $-0.3000$ | $93.0896$ | **Vulnerable:** Heavily pulled upward by extreme smog spikes. |
| **$L_1$ (Robust LP)** | $-0.2389$ | $77.7372$ | **Stable:** Unaffected by spikes, tracks the true dense data trend. |
| **Huber Regression** | $-0.2392$ | $77.9293$ | **Optimal:** Perfectly mirrors the robust $L_1$ trend via a smooth derivative. |

### Visual Insights
The execution of the project generates a dual-plot comprehensive dashboard:
* **Fitted Lines (Left):** Visually demonstrates how the $L_2$ line shifts away from the high-density standard atmospheric cluster to accommodate extreme values, while $L_1$ and Huber stay perfectly aligned with the baseline environmental trend.
* **Prediction Residuals (Right):** Explores the error trade-offs. Inside anomalous data bands, $L_2$ minimizes maximum errors at the cost of overall baseline degradation, whereas $L_1$ and Huber maintain superior baseline accuracy by allowing large, linear residual deviations on extreme points.

---

## Implementation Details

* **$L_1$ Solver:** Solved via `scipy.optimize.linprog` using the high-performance C++ underlying `HiGHS` solver method. The non-smooth absolute values are successfully mapped into an expanded block constraint design matrix.
* **Huber Solver:** Implemented using `scipy.optimize.minimize` running the `BFGS` algorithm with an adjustable tuning hyperparameter ($\delta = 15.0$) to seamlessly balance quadratic and linear error regions.
* **Defensive Data Pipeline:** Includes a localized parsing function utilizing `np.genfromtxt` to cleanly filter out non-numeric missing entries (`'NA'`) and features a robust synthetic fallback generator to ensure continuity if local resources are altered.

---

## Requirements & Environment

* **Runtime:** Python 3.7+
* **Dependencies:** NumPy, SciPy, Matplotlib

To install the required environment packages locally:

```bash
pip install -r requirements.txt