"""
L1 Regression for robust parameter estimation with outliers.

We solve the L1 minimization problem (min ||Ax - y||_1) as a linear program
since the absolute value isn't differentiable. Compare with L2 to show robustness.
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from pathlib import Path
import time


def l1_regression(A, y):
    """
    Solve L1 regression: min ||Ax - y||_1
    
    We reformulate as an LP with slack variables s:
    - minimize: sum(s)
    - s.t.: Ax - y <= s, y - Ax <= s, s >= 0
    
    At optimum, s_i = |residual_i|.
    """
    m, n = A.shape

    # Only the slack variables are penalized.
    c = np.concatenate([np.zeros(n), np.ones(m)])

    # Constraints for the absolute residuals.
    I_m = np.eye(m)

    G_upper = np.hstack([A, -I_m])
    G_lower = np.hstack([-A, -I_m])

    G = np.vstack([G_upper, G_lower])
    h = np.concatenate([y, -y])

    bounds_x = [(None, None)] * n
    bounds_s = [(0, None)] * m
    bounds = bounds_x + bounds_s

    result = linprog(c, A_ub=G, b_ub=h, bounds=bounds, method='highs')

    if not result.success:
        raise ValueError(f"LP solver failed: {result.message}")

    x_opt = result.x[:n]
    s_opt = result.x[n:]

    return x_opt, s_opt


def l2_regression(A, y):
    """Standard least-squares: min ||Ax - y||_2^2. For comparison."""
    x_opt, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return x_opt


def run_outlier_sweep(m=50, outlier_counts=(0, 5, 10, 15, 20), seed=42):
    """Plot how L1 and L2 errors change as we add more outliers."""
    l1_errors = []
    l2_errors = []

    for n_outliers in outlier_counts:
        A, y, _, x_true, _, _ = generate_corrupted_data(m, n_outliers, seed=seed)
        x_l1, _ = l1_regression(A, y)
        x_l2 = l2_regression(A, y)

        l1_errors.append(np.linalg.norm(x_l1 - x_true))
        l2_errors.append(np.linalg.norm(x_l2 - x_true))

    plt.figure(figsize=(7, 4.5))
    plt.plot(outlier_counts, l1_errors, 'o-', lw=2.5, label='L1 error')
    plt.plot(outlier_counts, l2_errors, 's-', lw=2.5, label='L2 error')
    plt.xlabel('Number of outliers')
    plt.ylabel('Parameter error')
    plt.title('Error vs. Outlier Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = Path(__file__).with_name('l1_vs_l2_outlier_sweep.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Outlier sweep: saved as 'l1_vs_l2_outlier_sweep.png'")
    print("  Outlier count   L1 error     L2 error")
    for count, l1_err, l2_err in zip(outlier_counts, l1_errors, l2_errors):
        print(f"  {count:<13} {l1_err:<11.4f} {l2_err:<.4f}")


# =============================================================================
# DEMONSTRATION: Synthetic dataset with heavy outlier corruption
# =============================================================================

def generate_corrupted_data(m=50, n_outliers=10, seed=42):
    """Generate synthetic affine data y=2t-1.5 with Gaussian noise + outliers."""
    np.random.seed(seed)

    x_true = np.array([2.0, -1.5])  # true slope and intercept
    t = np.linspace(0, 5, m)
    A = np.column_stack([t, np.ones(m)])

    # clean data + small noise
    y_clean = A @ x_true
    noise = 0.3 * np.random.randn(m)
    y = y_clean + noise

    # add large outliers
    outlier_idx = np.random.choice(m, n_outliers, replace=False)
    outlier_magnitudes = np.random.uniform(8, 15, n_outliers)
    outlier_signs = np.random.choice([-1, 1], n_outliers)
    y[outlier_idx] += outlier_magnitudes * outlier_signs

    return A, y, y_clean, x_true, outlier_idx, t


def main():
    """Run the comparison: fit data with both L1 and L2."""
    m = 50
    n_outliers = 10
    A, y, y_clean, x_true, outlier_idx, t = generate_corrupted_data(m, n_outliers)

    # solve L1 (robust)
    start = time.time()
    x_l1, s_l1 = l1_regression(A, y)
    t_l1 = time.time() - start

    # solve L2 (not robust)
    start = time.time()
    x_l2 = l2_regression(A, y)
    t_l2 = time.time() - start

    # compute errors
    err_l1 = np.linalg.norm(x_l1 - x_true)
    err_l2 = np.linalg.norm(x_l2 - x_true)

    print("=" * 60)
    print("L1 vs L2 Regression (with outliers)")
    print("=" * 60)
    print(f"\nDataset: {m} points, {n_outliers} outliers ({100*n_outliers/m:.0f}% corruption)")
    print(f"\n{'Parameter':<12} {'True':<10} {'L1':<12} {'L2':<12}")
    print("-" * 46)
    print(f"{'Slope':<12} {x_true[0]:<10.4f} {x_l1[0]:<12.4f} {x_l2[0]:<12.4f}")
    print(f"{'Intercept':<12} {x_true[1]:<10.4f} {x_l1[1]:<12.4f} {x_l2[1]:<12.4f}")
    print(f"\nParameter error: L1={err_l1:.4f}, L2={err_l2:.4f}")
    print(f"L1 is {err_l2/err_l1:.1f}x more accurate")
    print(f"Solve time: L1={t_l1*1000:.1f}ms, L2={t_l2*1000:.1f}ms")
    print("=" * 60)

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
           color='darkorange', alpha=0.8, label='L2 |residuals|')

    # Highlight outlier indices
    for idx in outlier_idx:
        ax.axvline(x=idx, color='gray', alpha=0.18, linewidth=4)

    ax.set_xlabel('Measurement index i', fontsize=12)
    ax.set_ylabel('|Residual|', fontsize=12)
    ax.set_title('Absolute Residuals per Measurement', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    main_plot_path = Path(__file__).with_name('l1_vs_l2_robust_regression.png')
    plt.savefig(main_plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  Plot saved as 'l1_vs_l2_robust_regression.png'")

    run_outlier_sweep()


if __name__ == "__main__":
    main()