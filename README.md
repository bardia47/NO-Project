# Robust Regression: L1 vs L2

Bardia Dorry (5866102), Sebastian James (6000546)

## Idea

When fitting a line to data, least squares (L2) gets thrown off by outliers because it penalizes big errors heavily. L1 regression (minimizing absolute residuals) is more robust because outliers don't blow up quadratically.

We solve the L1 problem as a linear program (LP) by introducing slack variables:
- minimize: Σ s_i
- subject to: |Ax - y| ≤ s and s ≥ 0

## Results

With 50 measurements and 10 outliers (20% corruption):

| Method | Slope | Intercept | Error |
|--------|-------|-----------|-------|
| True | 2.0000 | -1.5000 | — |
| L1 | 1.9652 | -1.5560 | 0.066 |
| L2 | 1.6619 | -0.6422 | 0.922 |

**L1 is 14× more accurate.** The plots show L1 stays close to the truth while L2 gets pulled toward the outliers.

As we add more outliers (0, 5, 10, 15, 20):

| Outliers | L1 Error | L2 Error |
|----------|----------|----------|
| 0 | 0.043 | 0.040 |
| 5 | 0.045 | 0.694 |
| 10 | 0.066 | 0.922 |
| 15 | 0.042 | 1.334 |
| 20 | 0.153 | 1.361 |

L1 stays stable while L2 falls apart.

## Implementation

We use `scipy.optimize.linprog` to solve the LP. The key insight is rewriting the non-smooth absolute value as a set of linear constraints.

## Usage

```bash
python main.py
```

Generates:
- `l1_vs_l2_robust_regression.png`: comparison plot with fitted lines and residuals
- `l1_vs_l2_outlier_sweep.png`: how error grows with outlier count

## Requirements

- Python 3.7+
- NumPy, SciPy, Matplotlib

```bash
pip install numpy scipy matplotlib
```

## Files

- `main.py`: Generates synthetic data, solves both L1 and L2, makes plots
