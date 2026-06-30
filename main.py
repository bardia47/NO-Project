import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# =============================================================================
# L1-Regression reformulated as a Linear Program (LP)
# =============================================================================
#
# Original Problem:  min ||Ax - y||_1  =  min  Σ |aᵢᵀx - yᵢ|
#
# The L1-norm is non-differentiable, so we introduce slack variables s ∈ ℝᵐ
# and reformulate as a smooth LP:
#
#   Decision Variables: z = [x; s]  where x ∈ ℝⁿ (model params), s ∈ ℝᵐ (slacks)
#
#   minimize    Σ sᵢ           (sum of slack variables)
#   subject to:  Ax - y <=  s   (residual upper bound)
#               -(Ax - y) <= s   (residual lower bound, i.e., y - Ax <= s)
#                s >= 0
#
# These constraints enforce: sᵢ >= |aᵢᵀx - yᵢ|
# Since we minimize Σsᵢ, at optimum: sᵢ = |aᵢᵀx - yᵢ|
# Therefore: min Σsᵢ  ≡  min ||Ax - y||₁
#
# =============================================================================


def l1_regression(A, y):
    """
    Solves the L1 regression problem: min ||Ax - y||_1
    by reformulating it as a Linear Program.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Design matrix (measurement/feature matrix)
    y : ndarray, shape (m,)
        Measurement vector (possibly corrupted by outliers)

    Returns
    -------
    x_opt : ndarray, shape (n,)
        Estimated model parameters
    s_opt : ndarray, shape (m,)
        Optimal slack variables (equal to absolute residuals at optimum)
    """
    m, n = A.shape

    # -------------------------------------------------------------------------
    # Step 1: Define the objective function
    # c = [0, 0, ..., 0, 1, 1, ..., 1]
    #       <--- n --->  <--- m --->
    # We only minimize the sum of slack variables s, not x directly.
    # -------------------------------------------------------------------------
    c = np.concatenate([np.zeros(n), np.ones(m)])

    # -------------------------------------------------------------------------
    # Step 2: Define inequality constraints  G @ z <= h
    #
    # Constraint 1:  Ax - s <= y    =>  [A | -I] * [x; s] <= y
    # Constraint 2: -Ax - s <= -y   =>  [-A | -I] * [x; s] <= -y
    #
    # Together these enforce: -s <= Ax - y <= s  =>  |Ax - y| <= s
    # -------------------------------------------------------------------------
    I_m = np.eye(m)

    G_upper = np.hstack([A, -I_m])    # Ax - s <= y
    G_lower = np.hstack([-A, -I_m])   # -Ax - s <= -y

    G = np.vstack([G_upper, G_lower])
    h = np.concatenate([y, -y])

    # -------------------------------------------------------------------------
    # Step 3: Define variable bounds
    # x: unbounded (free variables for model parameters)
    # s: non-negative (s >= 0, since they represent absolute values)
    # -------------------------------------------------------------------------
    bounds_x = [(None, None)] * n   # x is free
    bounds_s = [(0, None)] * m      # s >= 0
    bounds = bounds_x + bounds_s

    # -------------------------------------------------------------------------
    # Step 4: Solve the LP using HiGHS solver
    # -------------------------------------------------------------------------
    result = linprog(c, A_ub=G, b_ub=h, bounds=bounds, method='highs')

    if not result.success:
        raise ValueError(f"LP solver failed: {result.message}")

    # Extract solutions
    x_opt = result.x[:n]   # model parameters
    s_opt = result.x[n:]   # slack variables = |residuals|

    return x_opt, s_opt


def l2_regression(A, y):
    """
    Standard Least-Squares regression (for comparison).
    Solves: min ||Ax - y||_2^2
    Highly sensitive to outliers due to quadratic penalty.
    """
    x_opt, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return x_opt


# =============================================================================
# DEMONSTRATION: Synthetic dataset with heavy outlier corruption
# =============================================================================

def generate_corrupted_data(m=50, n_outliers=10, seed=42):
    """
    Generate a synthetic affine regression dataset with outliers.

    Parameters
    ----------
    m : int
        Number of measurements
    n_outliers : int
        Number of data points corrupted by large outliers
    seed : int
        Random seed for reproducibility

    Returns
    -------
    A : design matrix (m x 2)
    y : corrupted measurement vector
    y_clean : clean measurement vector (ground truth)
    x_true : true parameters
    outlier_idx : indices of outlier points
    """
    np.random.seed(seed)

    # True model: y = 2.0 * t - 1.5  (slope=2.0, intercept=-1.5)
    x_true = np.array([2.0, -1.5])

    # Generate uniformly spaced input points
    t = np.linspace(0, 5, m)

    # Design matrix for affine model: [t, 1]
    A = np.column_stack([t, np.ones(m)])

    # Clean measurements + small Gaussian noise
    y_clean = A @ x_true
    noise = 0.3 * np.random.randn(m)
    y = y_clean + noise

    # Inject large outliers (corrupt ~20% of data)
    outlier_idx = np.random.choice(m, n_outliers, replace=False)
    outlier_magnitudes = np.random.uniform(8, 15, n_outliers)
    outlier_signs = np.random.choice([-1, 1], n_outliers)
    y[outlier_idx] += outlier_magnitudes * outlier_signs

    return A, y, y_clean, x_true, outlier_idx, t


def main():
    # --- Generate data ---
    m = 50
    n_outliers = 10
    A, y, y_clean, x_true, outlier_idx, t = generate_corrupted_data(m, n_outliers)

    # --- Solve L1 regression (robust) ---
    x_l1, s_l1 = l1_regression(A, y)

    # --- Solve L2 regression (non-robust, for comparison) ---
    x_l2 = l2_regression(A, y)

    # --- Print results ---
    print("=" * 65)
    print("  ROBUST PARAMETER ESTIMATION: L1 vs. L2 Regression")
    print("=" * 65)
    print(f"\n  Dataset: {m} measurements, {n_outliers} outliers "
          f"({100*n_outliers/m:.0f}% corruption)")
    print(f"\n  {'Parameter':<14} {'True':<10} {'L1 (Robust)':<14} {'L2 (Least Sq.)':<14}")
    print("  " + "-" * 52)
    print(f"  {'Slope':<14} {x_true[0]:<10.4f} {x_l1[0]:<14.4f} {x_l2[0]:<14.4f}")
    print(f"  {'Intercept':<14} {x_true[1]:<10.4f} {x_l1[1]:<14.4f} {x_l2[1]:<14.4f}")

    # Compute estimation errors
    err_l1 = np.linalg.norm(x_l1 - x_true)
    err_l2 = np.linalg.norm(x_l2 - x_true)
    print(f"\n  L1 parameter error (||x_l1 - x_true||): {err_l1:.6f}")
    print(f"  L2 parameter error (||x_l2 - x_true||): {err_l2:.6f}")
    print(f"  L1 improvement factor: {err_l2 / err_l1:.1f}x more accurate")

    print(f"\n  Optimal L1 objective (Σ|residuals|): {np.sum(s_l1):.4f}")
    print("=" * 65)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fitted lines comparison
    ax = axes[0]
    t_plot = np.linspace(0, 5, 200)

    # Data points
    mask = np.ones(m, dtype=bool)
    mask[outlier_idx] = False
    ax.scatter(t[mask], y[mask], c='steelblue', alpha=0.7,
               edgecolors='k', linewidths=0.5, label='Inliers', zorder=3)
    ax.scatter(t[outlier_idx], y[outlier_idx], c='red', marker='x',
               s=120, linewidths=2, label='Outliers', zorder=4)

    # Fitted lines
    ax.plot(t_plot, x_true[0]*t_plot + x_true[1],
            'g--', lw=2.5, label=f'True: y={x_true[0]}t + ({x_true[1]})')
    ax.plot(t_plot, x_l1[0]*t_plot + x_l1[1],
            'b-', lw=2.5, label=f'L1: y={x_l1[0]:.3f}t + ({x_l1[1]:.3f})')
    ax.plot(t_plot, x_l2[0]*t_plot + x_l2[1],
            'r-', lw=2, label=f'L2: y={x_l2[0]:.3f}t + ({x_l2[1]:.3f})')

    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('L1 (Robust) vs L2 (Least Squares) Regression', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute residuals comparison
    ax = axes[1]
    residuals_l1 = np.abs(A @ x_l1 - y)
    residuals_l2 = np.abs(A @ x_l2 - y)

    indices = np.arange(m)
    bar_width = 0.4
    ax.bar(indices - bar_width/2, residuals_l1, bar_width,
           color='steelblue', alpha=0.8, label='L1 |residuals|')
    ax.bar(indices + bar_width/2, residuals_l2, bar_width,
           color='salmon', alpha=0.8, label='L2 |residuals|')

    # Highlight outlier indices
    for idx in outlier_idx:
        ax.axvline(x=idx, color='red', alpha=0.15, linewidth=4)

    ax.set_xlabel('Measurement index i', fontsize=12)
    ax.set_ylabel('|Residual|', fontsize=12)
    ax.set_title('Absolute Residuals per Measurement', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('l1_vs_l2_robust_regression.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  Plot saved as 'l1_vs_l2_robust_regression.png'")


if __name__ == "__main__":
    main()