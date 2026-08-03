"""Robust Regression Comparison: L1 (LP) vs L2 (OLS) vs Huber (QP)

REAL-WORLD DATASET: Cumulative wind speed (Iws) as the primary feature and PM2.5
    concentration as the target variable to compare L1 fitting loss, L2, and
    Huber loss.
"""

import sys
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


def create_features(x):
    """Creates non-linear features to increase the dimensionality of the problem (n=6)

    and fit the non-linear atmospheric decay trend.
    Features: 1/(a_k + x) for a_k in {1, 10, 20, 50, 100} plus intercept
    column.
    """
    a_k = [1, 10, 20, 50, 100]
    features = [1 / (a + x) for a in a_k]
    features.append(np.ones_like(x))
    return np.column_stack(features)


def l1_regression(A, y):
    """Solves L1 fitting loss by formulating it as a Linear Program (LP) in CVXPY.

    Parameters:
    -----------
    A : ndarray
        Design matrix of shape (num_samples, num_features).
    y : ndarray
        Target variable vector of shape (num_samples,).

    Returns:
    --------
    ndarray
        Optimal basis weight coefficients (n=6).
    """
    num_features = A.shape[1]
    x = cp.Variable(num_features)

    # Objective: Minimize L1 norm of residuals
    objective = cp.Minimize(cp.norm1(A @ x - y))
    prob = cp.Problem(objective)
    prob.solve()

    return x.value


def l2_regression(A, y):
    """Computes standard Ordinary Least Squares (OLS) via L2 minimization in CVXPY.

    Parameters:
    -----------
    A : ndarray
        Design matrix of shape (num_samples, num_features).
    y : ndarray
        Target variable vector of shape (num_samples,).

    Returns:
    --------
    ndarray
        Optimal basis weight coefficients vulnerable to outlier distortions.
    """
    num_features = A.shape[1]
    x = cp.Variable(num_features)

    # Objective: Minimize L2 norm (sum of squares) of residuals
    objective = cp.Minimize(cp.sum_squares(A @ x - y))
    prob = cp.Problem(objective)
    prob.solve()

    return x.value


def huber_regression(A, y, delta=15.0):
    """Solves robust Huber regression using a Quadratic Program (QP) formulation in CVXPY.

    Parameters:
    -----------
    A : ndarray
        Design matrix of shape (num_samples, num_features).
    y : ndarray
        Target variable vector of shape (num_samples,).
    delta : float, optional
        Threshold separating L2 quadratic loss from L1 linear loss. Defaults to
        15.0.

    Returns:
    --------
    ndarray
        Optimal robust basis weight coefficients.
    """
    num_samples, num_features = A.shape

    # Decision variables for QP formulation
    x = cp.Variable(num_features)
    u = cp.Variable(num_samples)  # L2 portion of error
    p = cp.Variable(num_samples)  # Positive L1 error slack
    q = cp.Variable(num_samples)  # Negative L1 error slack

    # Objective: 0.5 * ||u||_2^2 + delta * sum(p + q)
    objective = cp.Minimize(0.5 * cp.sum_squares(u) + delta * cp.sum(p + q))

    # Constraints for exact Huber equivalence
    constraints = [A @ x - y == u + p - q, p >= 0, q >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()  # Solved via QP solver (OSQP / ECOS)

    return x.value


# =============================================================================
# DATA LOADING
# =============================================================================


def load_environmental_data(file_path="pollution.csv"):
    """Loads historical Beijing environmental data to evaluate robust loss functions.

    Extracts cumulative wind speed (Iws) as the primary feature and PM2.5
    concentration as the target variable.

    Parameters:
    -----------
    file_path : str, optional
        Path to the local dataset file. Defaults to 'pollution.csv'.

    Returns:
    --------
    A : ndarray
        The 6D design matrix of shape (N, 6) containing non-linear features and
        intercept.
    pm25 : ndarray
        The target vector of shape (N,) containing PM2.5 pollution levels.
    outlier_idx : ndarray
        Indices of data points flagged as outliers based on the top 20% L2
        residuals.
    wind_speed : ndarray
        The raw wind speed values (Iws) of shape (N,).
    """
    print(f"Loading dataset from local file path: '{file_path}'...")
    try:
        raw_data = np.genfromtxt(
            file_path, delimiter=",", skip_header=1, usecols=(10, 5)
        )
        clean_mask = ~np.isnan(raw_data).any(axis=1)
        data = raw_data[clean_mask]
        wind_speed = data[:100, 0]
        pm25 = data[:100, 1]
    except Exception as e:
        print(f"Local file not found or corrupted: {e}")
        print("Falling back to synthetic matrix for continuity...")
        np.random.seed(10)
        wind_speed = np.array([
            1.79,
            4.92,
            9.84,
            12.97,
            18.21,
            2.34,
            5.71,
            14.22,
            22.11,
            1.12,
            30.45,
            41.22,
            1.55,
            3.82,
            0.99,
            145.2,
            110.1,
            118.5,
            122.4,
            130.0,
        ])
        pm25 = np.array([
            129.0,
            145.0,
            110.0,
            95.0,
            80.0,
            150.0,
            120.0,
            75.0,
            50.0,
            180.0,
            35.0,
            22.0,
            165.0,
            138.0,
            195.0,
            12.1,
            15.3,
            1.5,
            4.0,
            5.1,
        ])

    # Construct the nonlinear design matrix (n=6)
    A = create_features(wind_speed)

    # Identify outliers based on baseline L2 fit
    x_l2_baseline = l2_regression(A, pm25)
    residuals = np.abs(A @ x_l2_baseline - pm25)
    outlier_idx = np.where(residuals > np.percentile(residuals, 80))[0]

    return A, pm25, outlier_idx, wind_speed


def benchmark_scaling(file_path="pollution.csv"):
    """Evaluates solver runtime scaling across increasing dataset sizes (m)."""
    sizes = [100, 500, 1000, 3000]
    print("\n" + "=" * 70)
    print("SOLVER SCALING BENCHMARK (Runtime vs Dataset Size m)")
    print("=" * 70)
    print(
        f"{'m (Samples)':<12} | {'L2 (ms)':<10} | {'L1 LP (ms)':<10} |"
        f" {'Huber QP (ms)':<10}"
    )
    print("-" * 55)

    try:
        raw_data = np.genfromtxt(
            file_path, delimiter=",", skip_header=1, usecols=(10, 5)
        )
        clean_mask = ~np.isnan(raw_data).any(axis=1)
        data = raw_data[clean_mask]
    except Exception:
        print("Dataset file not available for full benchmark scale.")
        return

    for m_size in sizes:
        if m_size > len(data):
            break
        wind_speed = data[:m_size, 0]
        pm25 = data[:m_size, 1]
        A = create_features(wind_speed)

        # Measure L2 Execution Time
        t0 = time.time()
        l2_regression(A, pm25)
        t_l2 = (time.time() - t0) * 1000

        # Measure L1 Execution Time
        t0 = time.time()
        l1_regression(A, pm25)
        t_l1 = (time.time() - t0) * 1000

        # Measure Huber Execution Time
        t0 = time.time()
        huber_regression(A, pm25, delta=15.0)
        t_huber = (time.time() - t0) * 1000

        print(
            f"{m_size:<12} | {t_l2:<10.2f} | {t_l1:<10.2f} | {t_huber:<10.2f}"
        )
    print("=" * 70 + "\n")
        

def main():
    # Load Dataset
    A, pm25, outlier_idx, wind_speed = load_environmental_data()
    m, num_features = A.shape

    # Fit models on Data and measure execution time
    start = time.time()
    x_l1 = l1_regression(A, pm25)
    t_l1 = time.time() - start

    start = time.time()
    x_l2 = l2_regression(A, pm25)
    t_l2 = time.time() - start

    start = time.time()
    x_huber = huber_regression(A, pm25, delta=15.0)
    t_huber = time.time() - start

    print("=" * 70)
    print(f"Number of Decision Variables (Weights): {num_features}")
    print("=" * 70)
    print(f"Execution Times (CVXPY Solvers):")
    print(
        f"L2: {t_l2*1000:.2f}ms | L1 (LP): {t_l1*1000:.2f}ms | Huber (QP):"
        f" {t_huber*1000:.2f}ms"
    )
    print("=" * 70)

    # --- Run Scaling Benchmark ---
    benchmark_scaling()

    # --- Visualization ---
    # Plot 1: Fitted Non-linear Curves
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))

    t_plot = np.linspace(wind_speed.min(), wind_speed.max(), 200)
    A_plot = create_features(t_plot)

    mask = np.ones(m, dtype=bool)
    mask[outlier_idx] = False

    ax1.scatter(
        wind_speed[mask],
        pm25[mask],
        c="steelblue",
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
        label="Standard Atmospheric Conditions",
        zorder=3,
    )

    ax1.scatter(
        wind_speed[outlier_idx],
        pm25[outlier_idx],
        c="red",
        marker="X",
        s=90,
        linewidths=1.5,
        label="Anomalies / Extreme Weather",
        zorder=4,
    )

    ax1.plot(
        t_plot,
        A_plot @ x_l2,
        "r-",
        lw=2.5,
        label="L2 (Least Squares) - Shifted by Spikes",
    )
    ax1.plot(
        t_plot,
        A_plot @ x_l1,
        "b-",
        lw=2.5,
        label="L1 Fitting Loss (Robust LP)",
    )
    ax1.plot(
        t_plot,
        A_plot @ x_huber,
        "g-",
        lw=2.5,
        label="Huber Loss (Robust QP)",
    )

    ax1.set_xlabel("Cumulative Wind Speed (Iws in m/s)", fontsize=11)
    ax1.set_ylabel(r"PM2.5 Concentration ($\mu g/m^3$)", fontsize=11)
    ax1.set_title(
        "Non-linear Robust Fitting: Wind Speed vs Pollution", fontsize=13
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig("robust fitting.png", dpi=300, bbox_inches="tight")

    # Plot 2: Absolute residuals on Data
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))

    residuals_l1 = np.abs(A @ x_l1 - pm25)
    residuals_l2 = np.abs(A @ x_l2 - pm25)
    residuals_huber = np.abs(A @ x_huber - pm25)

    indices = np.arange(m)
    bar_width = 0.25

    ax2.bar(
        indices - bar_width,
        residuals_l1,
        bar_width,
        color="blue",
        alpha=0.6,
        label="L1 Error",
    )
    ax2.bar(
        indices,
        residuals_huber,
        bar_width,
        color="green",
        alpha=0.6,
        label="Huber Error",
    )
    ax2.bar(
        indices + bar_width,
        residuals_l2,
        bar_width,
        color="red",
        alpha=0.6,
        label="L2 Error",
    )

    for idx in outlier_idx:
        ax2.axvline(x=idx, color="gray", alpha=0.12, linewidth=4)

    ax2.set_xlabel("Data Point Index", fontsize=11)
    ax2.set_ylabel("|Residual Error|", fontsize=11)
    ax2.set_title("Prediction Residuals Comparison", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("residual error.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()