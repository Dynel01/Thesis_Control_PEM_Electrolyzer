import numpy as np
import matplotlib.pyplot as plt
from PEMWE_Parameters import *
from Resistance_Calc import L_pem_cm, R_others

R = 8.314
F = 96485
z = 2
temp_c = 55
T = temp_c + 273.15
def p_sat(T):
    """
    Saturation vapour pressure [bar]
    Arden Buck equation, valid for liquid water T > 0°C
    """
    T_c = T - 273.15
    p_hpa = 6.1121 * np.exp((18.678 - T_c / 234.5) * (T_c / (257.14 + T_c)))
    return p_hpa / 1000  # hPa to bar

def V_rev_TP(T, p_cathode,p_anode):
    # Partial pressures
    p_H2O = p_sat(T)
    p_H2  = max(1e-4,p_cathode - p_H2O)
    p_O2  = max(1e-4,p_anode   - p_H2O)
    # Temperature-dependent reversible voltage (Gibbs-based fit)
    V_rev_T = 1.5184 - 1.5421e-3*T + 9.523e-5*T*np.log(T) + 9.84e-8*T**2

    # Nernst correction for pressure
    V_rev_nernst = ((R * T) / (z * F)) * np.log((p_H2 * (p_O2**0.5)) / p_H2O)

    V_total_rev = V_rev_T + V_rev_nernst
    return V_total_rev
    

T_range = np.linspace(10 + 273, 80 + 273, 50)  # K
p_range = np.linspace(1, 100, 50)                       # bar cathode pressure


# Plotting effect of Nernst correction
V_rev_T_plot = [1.5184 - 1.5421e-3*T + 9.523e-5*T*np.log(T) + 9.84e-8*T**2 for T in T_range]

p_range = np.linspace(5, 100, 100)
temps   = [328.15, 338.15, 353.15]
labels  = ['55°C', '65°C', '80°C']
colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']

def j0_T(j0_ref, Ea, T):
    """
    Temperature dependent exchange current density [A/cm²]
    T : temperature [K]
    """
    return j0_ref * np.exp((Ea / R) * (1/T0_ref - 1/T))

def eta_act(T, j):
    """
    Activation overpotential [V]
    T : temperature [K]
    j : current density [A/cm²]
    """
    j0_ano = j0_T(j0_anode_ref, Ea_anode, T)
    j0_cat = j0_T(j0_cathode_ref, Ea_cathode, T)

    eta_anode   = (R * T / F) * np.arcsinh(j / (2 * j0_ano))
    eta_cathode = (R * T / F) * np.arcsinh(j / (2 * j0_cat))
    return eta_anode + eta_cathode

T_test = T_nom
j_test = j_nom

j_range = np.linspace(0.01, j_max, 50)
temps1   = [T_min, T_nom, T_max]
labels1  = ['40°C', '55°C', '80°C']
colors1  = ['#1f77b4', '#ff7f0e', '#2ca02c']

def sigma_mem(T):
    """
    Membrane ionic conductivity [S/cm]
    Olivier et al. Eq. 20
    Uses lambda_m from Springer et al. for fully hydrated Nafion
    """
    return (0.005139 * lambda_m - 0.03261) * np.exp(1268 * (1/303 - 1/T))

# ── Ohmic Parameters (Olivier et al. 2017, Eq. 97) ────────────────────────

def r_eq(T):
    """
    Temperature dependent equivalent ohmic resistance [Ω]
    Lumps membrane + contact + bipolar plate resistances
    Olivier et al. 2017, Eq. 97
    T in Kelvin
    """
    R_eql= -2e-5 * T + 0.0078
    return R_eql


def eta_ohm(T, j):
    """
    Ohmic overpotential [V]
    Olivier et al. 2017, Eq. 97
    R_eq is treated as total ohmic resistance
    (membrane + contacts + bipolar plates)
    Note: Olivier's fitted R_eq absorbs all ohmic contributions
    T : temperature [K]
    j : current density [A/cm²]
    """
    I_cell = j * A_cell    # A
    R_ohm= L_pem_cm/(sigma_mem(T)*A_cell)+R_others
    return R_ohm * I_cell

def V_cell(T, j, p_cathode, p_anode):
    """
    Full cell voltage [V]
    T         : temperature [K]
    j         : current density [A/cm²]
    p_cathode : cathode pressure [bar]
    p_anode   : anode pressure [bar]
    """
    return V_rev_TP(T, p_cathode, p_anode) + eta_act(T, j) + eta_ohm(T, j)

# Polarization curve

j_range1 = np.linspace(0.01, j_max, 50)
temps2   = [T_min, T_nom, T_max]
labels2  = ['40°C', '55°C', '80°C']
colors2  = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Individual Overpotential
j_range = np.linspace(0.01, j_max, 50)
temp3   = [T_nom]
label3  = [ 'Activation', 'Ohmic', 'Total']
color3  = ['#1f77b4']


# Temperature-Dependent Thermoneutral Voltage
def V_tn(T):
    """Thermoneutral voltage [V], T in Kelvin"""
    return 1.4756 + 2.252e-4 * T + 1.52e-8 * T**2
            

test_temps_c = (40, 55, 80)


if __name__ == "__main__":
    print("--- 210 MW Hub Simulation Starting ---")
    
    print("── p_sat across operating range ──")
    for t_c in [60, 65, 70, 75, 80]:
        T = t_c + 273.15
        print(f"  {t_c}°C -> p_sat = {p_sat(T):.5f} bar")

    print("\n── V0(T) alone ──")
    for t_c in [60, 65, 70, 75, 80]:
        T = t_c + 273.15
        V0 = (1.5184 - 1.5421e-3*T + 9.523e-5*T*np.log(T) + 9.84e-8*T**2)
        print(f"  {t_c}°C -> V0 = {V0:.4f} V")

    print("\n── V_rev with p_cathode=30 bar, p_anode=1.01325 bar ──")
    for t_c in [60, 65, 70, 75, 80]:
        T = t_c + 273.15
        print(f"  {t_c}°C -> V_rev = {V_rev_TP(T, p_cathode, p_anode):.4f} V")
        
    T_grid, p_grid = np.meshgrid(T_range, p_range)

    # Calculate V_rev over grid
    V_grid = np.vectorize(lambda T, p: V_rev_TP(T, p, p_anode))(T_grid, p_grid)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot( projection='3d')

    surf = ax.plot_surface(
        T_grid - 273.15,   # convert back to °C for readability
        p_grid,
        V_grid,
        cmap='viridis',
        alpha=0.9
    )

    ax.set_xlabel('Temperature (°C)', fontsize=11, labelpad=10)
    ax.set_ylabel('Cathode Pressure (bar)', fontsize=11, labelpad=10)
    ax.set_zlabel('V_rev (V)', fontsize=11, labelpad=10)
    ax.set_title('Reversible Potential V_rev(T, P)', fontsize=13)

    fig.colorbar(surf, ax=ax, shrink=0.5, label='V_rev (V)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(T_range - 273.15, V_rev_T_plot,   label='V0(T) — standard potential', 
            linewidth=2, linestyle='--')
    plt.plot(T_range - 273.15, V_rev_TP(T_range, p_cathode, p_anode), label='V_rev(T) — with Nernst correction', 
            linewidth=2)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('Effect of Nernst Correction on Reversible Potential', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for T, lab, col in zip(temps, labels, colors):
        V = [V_rev_TP(T, p, p_anode) for p in p_range]
        plt.plot(p_range, V, label=lab, linewidth=2, color=col)

    plt.xlabel('Cathode Pressure (bar)', fontsize=12)
    plt.ylabel('V_rev (V)', fontsize=12)
    plt.title('Reversible Potential vs Cathode Pressure', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"── j0 temperature dependence ──")
    for t_c in [25, 30, 35, 40, 45, 50, 55]:
        T = t_c + 273.15
        print(f"  {t_c}°C → j0_anode = {j0_T(j0_anode_ref, Ea_anode, T):.3e}"
            f"  j0_cathode = {j0_T(j0_cathode_ref, Ea_cathode, T):.4f}")

    print(f"\n── Voltage breakdown at T_nom=55°C, j_nom=0.92 A/cm² ──")
    print(f"  V_rev    : {V_rev_TP(T_test, p_cathode, p_anode):.4f} V")
    print(f"  eta_act  : {eta_act(T_test, j_test):.4f} V")
    print(f"    anode  : {(R*T_test/F)*np.arcsinh(j_test/(2*j0_T(j0_anode_ref, Ea_anode, T_test))):.4f} V")
    print(f"    cathode: {(R*T_test/F)*np.arcsinh(j_test/(2*j0_T(j0_cathode_ref, Ea_cathode, T_test))):.4f} V")

    print(f"  eta_ohm  : {eta_ohm(T_nom, j_nom):.4f} V")
    print(f"    R_mem  : {L_pem_cm/(sigma_mem(T_nom)*A_cell):.6f} Ω")
    print(f"    R_other: {R_others:.6f} Ω")
    print(f"    R_Fit: {r_eq(T_nom):.6f} Ω")
    print(f"    I_cell : {j_nom*A_cell:.4f} A")
    print(f"  ───────────────────────────────")
    print(f"  V_cell   : {V_cell(T_nom, j_nom, p_cathode, p_anode):.4f} V")
    
    plt.figure(figsize=(8, 5))
    for T1, lab1, col1 in zip(temps1, labels1, colors1):
        eta = [eta_act(T1, j) for j in j_range]
        plt.plot(j_range, eta, label=lab1, linewidth=2, color=col1)

    plt.xlabel('Current Density (A/cm²)', fontsize=12)
    plt.ylabel('η_act (V)', fontsize=12)
    plt.title('Activation Overpotential vs Current Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for T2, lab2, col2 in zip(temps2, labels2, colors2):
        V = [V_cell(T2, j, p_cathode, p_anode) for j in j_range1]
        plt.plot(j_range1, V, label=lab2, linewidth=2, color=col2)
    plt.xlabel('Current Density (A/cm²)', fontsize=12)
    plt.ylabel('V_cell (V)', fontsize=12)
    plt.title('PEM Electrolyzer Polarization Curve', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for T3, lab3, col3 in zip(temp3, label3, color3):
        eta = [eta_act(T3, j) for j in j_range]
        ohm= [eta_ohm(T3, j) for j in j_range]
        tot= [V_cell(T3,j,p_cathode, p_anode)- V_rev_TP(T3, p_cathode, p_anode) for j in j_range]
        plt.plot(j_range, eta, label='Activation', linewidth=2, color='r')
        plt.plot(j_range, ohm, label='Ohmic', linewidth=2, color='g')
        plt.plot(j_range, tot, label='Total', linewidth=2, color=col3)
        plt.xlabel('Current Density (A/cm²)', fontsize=12)
    plt.ylabel('Overpotentials (V)', fontsize=12)
    plt.title('Individual Overpotential vs Current Density at 55°C', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test_temps_c, [V_tn(tc + 273.15) for tc in test_temps_c], lw=2)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Thermoneutral Voltage (V)', fontsize=12)
    plt.title('TNV with Temperature', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    j_range2 = np.linspace(0.1, 1.2, 50)
    T_mesh,J_mesh = np.meshgrid(T_range, j_range2)
    V_eff = np.zeros((len(T_range), len(j_range2)))
    for a in range(len(T_range)):
        for b in range(len(j_range2)):
            current_density = J_mesh[a,b]
            temperature = T_mesh[a,b]
            voltage = V_cell(temperature, current_density, p_cathode, p_anode)
            tnv= V_tn(temperature)
            
            #Calculate Voltage Efficiency
            
            V_eff[a,b] = tnv/voltage*100 
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot( projection='3d')

    surf = ax.plot_surface(
        T_mesh - 273.15,   # convert back to °C for readability
        J_mesh,
        V_eff,
        cmap='viridis',
        alpha=0.9
    )

    ax.set_xlabel('Temperature (°C)', fontsize=11, labelpad=10)
    ax.set_ylabel('Current Density (A/cm²)', fontsize=11, labelpad=10)
    ax.set_zlabel('Voltage Efficiency (%)', fontsize=11, labelpad=10)
    ax.set_title('Voltage Efficiency(V_eff(T, J))', fontsize=13)

    fig.colorbar(surf, ax=ax, shrink=0.5, label='V_eff (%)')
    plt.tight_layout()
    plt.show()
            
            
            
