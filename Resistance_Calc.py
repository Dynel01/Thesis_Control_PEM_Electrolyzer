import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- 1. Constants and Table 6 Data [cite: 349, 1324] ---
A_cell = 130          # cm2
lambda_m = 14         # Assumed hydration constant
temp_table_c = np.array([25, 30, 35, 40, 45, 50])
temp_table_k = temp_table_c + 273.15
# R_eq values from Table 6 
r_table_ohm = np.array([1.80e-3, 1.70e-3, 1.60e-3, 1.50e-3, 1.40e-3, 1.30e-3])

# --- 2. Conductivity Function (Equation 20) [cite: 495] ---
def calc_sigma(T, lam):
    # Returns conductivity in S/cm
    
    term1 = 0.005139 * lam - 0.0326
    term2 = np.exp(1268 * (1/303.15 - 1/T))
    return term1 * term2

# --- 3. The Physics Fit (25-50°C) ---
# Goal: R_total = L_pem * (1 / (sigma * A)) + R_others
sigma_table = calc_sigma(temp_table_k, lambda_m)
X_fit = 1 / (sigma_table * A_cell)
Y_fit = r_table_ohm

slope, intercept, r_value, p_value, std_err = linregress(X_fit, Y_fit)

L_pem_cm = slope
R_others = intercept

# --- 4. Extrapolation and Comparison (20-80°C) ---
T_range_c = np.linspace(20, 80, 100)
T_range_k = T_range_c + 273.15

# a) Linear Model from Paper (Equation 97) 
R_linear = -2e-5 * T_range_k + 0.0078

# b) Physics Model (Our solved constants)
sigma_range = calc_sigma(T_range_k, lambda_m)
R_physics = L_pem_cm / (sigma_range * A_cell) + R_others

# --- 5. Visualization ---
if __name__=='__main__':
    plt.figure(figsize=(10, 6))
    plt.plot(T_range_c, R_linear * 1e3, '--', label='Linear Fit (Paper Eq. 97)', color='gray')
    plt.plot(T_range_c, R_physics * 1e3, label='Physics-Based Extrapolation', color='blue', linewidth=2)
    plt.scatter(temp_table_c, r_table_ohm * 1e3, color='red', label='Table 6 Data Points', zorder=5)

    plt.xlabel('Temperature (°C)')
    plt.ylabel('Total Resistance (mΩ)')
    plt.title('Comparison: Empirical Linear Fit vs. Physical Membrane Fit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"--- Solved Physical Constants ---")
    print(f"L_PEM: {L_pem_cm*1e4:.2f} um")
    print(f"R_others: {R_others*1e3:.4f} mOhm")
    print(f"R-squared: {r_value**2:.4f}")