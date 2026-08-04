# Robust Parameter Estimation: L1 LP vs L2 OLS vs Huber QP

This repository contains the Python implementation and experimental benchmark for comparing **Ordinary Least Squares ($L_2$)**, **$L_1$-norm fitting loss (Linear Program)**, and **Huber Loss (Quadratic Program)** parameter estimation models on [atmospheric environmental data from Beijing](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv).

This work was developed as part of the Numerical Optimization project module at the University of Freiburg.

---

## Project Overview

Standard Least Squares ($L_2$) regression is highly sensitive to leverage outliers because it penalizes residual errors quadratically. This project evaluates two convex, robust alternatives:
1. **$L_1$ Fitting Loss (LP):** Reformulated as a smooth Linear Program using non-negative slack variables $s \in \mathbb{R}^m$.
2. **Huber Loss (QP):** Reformulated as an exact, convex Quadratic Program by splitting residuals into $L_2$ quadratic components $u \in \mathbb{R}^m$ and $L_1$ linear slack components $p, q \in \mathbb{R}^m$.

To move beyond simplified 2-variable linear fits, the model maps cumulative wind speed ($Iws$) into a **6-dimensional non-linear feature space ($n=6$)** using multi-scale fractional basis functions $\frac{1}{a_k + x}$ ($a_k \in \{1, 10, 20, 50, 100\}$) plus a bias intercept.

---

## Repository Structure

```text
.
├── main.py              # Main evaluation script (model fitting, residual plots, scaling benchmark)
├── pollution.csv        # Beijing atmospheric dataset (Iws vs PM2.5)
├── robust fitting.png   # Output plot: Non-linear regression fits (n=6)
├── residual error.png   # Output plot: Residual error distributions across data points
└── README.md            # Repository documentation
```

---

## Requirements & Installation

The project uses Python 3.8+ and relies on the `CVXPY` convex optimization modeling framework routed through C++ solver backends (`OSQP`, `ECOS`, `HiGHS`).

1. Clone the repository:
   git clone https://github.com/bardia47/NO-Project.git
   cd NO-Project

2. Install required dependencies:
   pip install numpy cvxpy matplotlib

---

## Execution Instructions

To run the complete parameter estimation pipeline, generate figures, and display solver execution runtimes, run:

```python
python main.py
```

### What happens during execution:
1. **Data Loading & Defensive Fallback:** The pipeline attempts to load `pollution.csv`. If the file is missing or unreadable, it gracefully falls back to a synthetic distribution without crashing.
2. **Feature Extraction:** Constructs the design matrix $A \in \mathbb{R}^{m \times 6}$ using the non-linear fractional basis functions.
3. **Model Fitting:** Solves $L_2$, $L_1$ LP, and Huber QP models via `CVXPY`.
4. **Solver Scaling Benchmark:** Runs a runtime comparison across dataset sizes $m \in \{100, 500, 1000, 3000\}$.
5. **Visualization:** Saves two high-resolution plots (`robust fitting.png` and `residual error.png`) locally and opens interactive plot windows.

---

## Optimization Formulations

### 1. $L_1$ Fitting Loss (Linear Program)
$$\min_{x \in \mathbb{R}^n, \, s \in \mathbb{R}^m} \sum_{i=1}^{m} s_i \quad \text{s.t.} \quad A_i x - s_i \le y_i, \quad -A_i x - s_i \le -y_i$$

### 2. Huber Loss (Quadratic Program)
$$\min_{x \in \mathbb{R}^n, u, p, q \in \mathbb{R}^m} \frac{1}{2} \|u\|_2^2 + \delta \sum_{i=1}^{m} (p_i + q_i) \quad \text{s.t.} \quad A x - y = u + p - q, \quad p \ge 0, \, q \ge 0$$
*(Default threshold $\delta = 15.0 \, \mathrm{\mu g/m^3}$)*

---

## Benchmark Results ($n=6$ Basis Features)

| Sample Size ($m$) | $L_2$ OLS (ms) | $L_1$ LP (ms) | Huber QP (ms) |
|:-----------------:|:--------------:|:-------------:|:-------------:|
| **100**           | 17.17          | 13.34         | 15.13         |
| **500**           | 15.05          | 20.19         | 26.40         |
| **1000**          | 24.58          | 36.44         | 32.11         |
| **3000**          | 27.47          | 81.50         | 64.93         |

### Key Observations:
* **$L_2$ Least Squares** is fast due to direct normal equation factorizations ($\mathcal{O}(m n^2 + n^3)$), but produces heavily biased parameter estimates under severe pollution spikes.
* **$L_1$ LP** provides strong robustness against leverage outliers, but its runtime scales sharply as $m$ increases due to introducing $m$ slack variables and $2m$ constraints.
* **Huber QP** achieves the robust fitting behavior of $L_1$ while scaling more efficiently at larger dataset sizes ($m=3000$) due to smooth quadratic gradient iterations in operator-splitting solvers like OSQP.

---

## Authors

* **Bardia Dorry** (Matriculation No.: 5866102) — *MSc Computer Science, University of Freiburg*
* **Sebastian James** (Matriculation No.: 6000546) — *MSc Embedded Systems, University of Freiburg*