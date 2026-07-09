"""
Robust Regression Comparison: L1 vs L2 vs Huber
Real-world scenario: Wind Energy Generation vs Wind Speed Forecast
"""

import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt
from pathlib import Path
import time

def l1_regression(A, y):
    """Solve L1 regression using Linear Programming."""
    m, n = A.shape
    c = np.concatenate([np.zeros(n), np.ones(m)])
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

    return result.x[:n]

def l2_regression(A, y):
    """Standard least-squares (L2)."""
    x_opt, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return x_opt

def huber_regression(A, y, delta=2.0):
    """
    Solve Huber regression using SciPy minimize.
    Huber loss combines L1 (for outliers) and L2 (for inliers).
    """
    def huber_loss(x):
        residuals = A @ x - y
        abs_r = np.abs(residuals)
        # Huber loss formula
        loss = np.where(abs_r <= delta,
                        0.5 * residuals**2,
                        delta * (abs_r - 0.5 * delta))
        return np.sum(loss)

    # Use L2 solution as a good initial guess
    x0 = l2_regression(A, y)
    result = minimize(huber_loss, x0, method='BFGS')
    return result.x

# =============================================================================
# DATASET: Wind Speed vs Generated Power (with sensor failures)
# =============================================================================

def get_wind_energy_data(m=50, n_outliers=8, seed=51):
    """
    Simulates wind speed (m/s) vs generated power (MW).
    Includes real-world anomalies like sensor freezing or grid curtailment.
    """
    np.random.seed(seed)

    # Wind speed forecast (X-axis): from 4 m/s to 15 m/s
    wind_speed = np.linspace(4, 15, m)
    A = np.column_stack([wind_speed, np.ones(m)])

    # True relationship (Slope and Intercept for this specific operating region)
    x_true = np.array([1.8, -5.0])

    # Normal operation (clean data + natural wind variance/noise)
    power_clean = A @ x_true
    noise = 1.2 * np.random.randn(m)
    power = power_clean + noise

    # Inject real-world outliers
    outlier_idx = np.random.choice(m, n_outliers, replace=False)

    for idx in outlier_idx:
        anomaly_type = np.random.choice(['sensor_freeze', 'power_surge'])
        if anomaly_type == 'sensor_freeze':
            # Turbine running, but sensor reports zero or near zero
            power[idx] = np.random.uniform(0, 2)
        else:
            # Data logging error causing massive false spikes
            power[idx] += np.random.uniform(15, 25)

    return A, power, x_true, outlier_idx, wind_speed

def main():
    m = 50
    n_outliers = 8
    A, y, x_true, outlier_idx, wind_speed = get_wind_energy_data(m, n_outliers)

    # 1. Fit models
    start = time.time()
    x_l1 = l1_regression(A, y)
    t_l1 = time.time() - start

    start = time.time()
    x_l2 = l2_regression(A, y)
    t_l2 = time.time() - start

    start = time.time()
    # Delta controls where it switches from L2 to L1 behavior
    x_huber = huber_regression(A, y, delta=2.5)
    t_huber = time.time() - start

    # 2. Compute Parameter Errors
    err_l1 = np.linalg.norm(x_l1 - x_true)
    err_l2 = np.linalg.norm(x_l2 - x_true)
    err_huber = np.linalg.norm(x_huber - x_true)

    print("=" * 70)
    print("Wind Energy Prediction: L1 vs L2 vs Huber Regression")
    print("=" * 70)
    print(f"\nDataset: {m} hours of data, {n_outliers} sensor anomalies ({100*n_outliers/m:.0f}% corruption)")
    print(f"\n{'Parameter':<12} {'Clean Trend':<15} {'L1':<12} {'L2':<12} {'Huber':<12}")
    print("-" * 65)
    print(f"{'Slope':<12} {x_true[0]:<15.4f} {x_l1[0]:<12.4f} {x_l2[0]:<12.4f} {x_huber[0]:<12.4f}")
    print(f"{'Intercept':<12} {x_true[1]:<15.4f} {x_l1[1]:<12.4f} {x_l2[1]:<12.4f} {x_huber[1]:<12.4f}")
    print(f"\nModel Error (Distance from clean trend):")
    print(f"L2 Error:    {err_l2:.4f} (Worst)")
    print(f"Huber Error: {err_huber:.4f} (Better)")
    print(f"L1 Error:    {err_l1:.4f} (Best for severe outliers)")
    print("=" * 70)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Plot 1: Fitted lines
    ax = axes[0]
    t_plot = np.linspace(4, 15, 200)

    # Data points
    mask = np.ones(m, dtype=bool)
    mask[outlier_idx] = False

    ax.scatter(wind_speed[mask], y[mask], c='steelblue', alpha=0.8,
               edgecolors='k', linewidths=0.5, label='Normal Operations', zorder=3)
    ax.scatter(wind_speed[outlier_idx], y[outlier_idx], c='red', marker='X',
               s=100, linewidths=1.5, label='Anomalies (Sensor/Grid Errors)', zorder=4)

    # Fitted lines
    ax.plot(t_plot, x_true[0]*t_plot + x_true[1], 'k--', lw=2, alpha=0.6, label='Clean Trend')
    ax.plot(t_plot, x_l1[0]*t_plot + x_l1[1], 'b-', lw=2.5, label='L1 (Robust)')
    ax.plot(t_plot, x_huber[0]*t_plot + x_huber[1], 'g-', lw=2.5, label='Huber (Balanced)')
    ax.plot(t_plot, x_l2[0]*t_plot + x_l2[1], 'r-', lw=2.5, label='L2 (Least Squares)')

    ax.set_xlabel('Forecasted Wind Speed (m/s)', fontsize=11)
    ax.set_ylabel('Generated Power (MW)', fontsize=11)
    ax.set_title('Wind Power Curve Estimation with Sensor Errors', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute residuals comparison
    ax = axes[1]
    residuals_l1 = np.abs(A @ x_l1 - y)
    residuals_l2 = np.abs(A @ x_l2 - y)
    residuals_huber = np.abs(A @ x_huber - y)

    indices = np.arange(m)
    bar_width = 0.25  # Thinner bars to fit 3 of them

    ax.bar(indices - bar_width, residuals_l1, bar_width, color='blue', alpha=0.6, label='L1 Error')
    ax.bar(indices, residuals_huber, bar_width, color='green', alpha=0.6, label='Huber Error')
    ax.bar(indices + bar_width, residuals_l2, bar_width, color='red', alpha=0.6, label='L2 Error')

    # Highlight outlier indices
    for idx in outlier_idx:
        ax.axvline(x=idx, color='gray', alpha=0.15, linewidth=5)

    ax.set_xlabel('Measurement Data Points (Hours)', fontsize=11)
    ax.set_ylabel('Absolute Prediction Error (MW)', fontsize=11)
    ax.set_title('Prediction Errors per Measurement', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    main_plot_path = Path(__file__).with_name('wind_energy_robust_regression.png')
    plt.savefig(main_plot_path, dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()