
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PEMWE_Parameters import *
from Electrochemical_Model import *
from Thermal_Model import *

N_stacks = 1 # We'll look at 1 stack first for clarity
a1 = -0.0034
a2 = -0.001711
a= a1*p_cathode+ a2
a_corrected = a * (25.4 / 84.81)
b= -1
c=1
load_percent= np.linspace(10, 100, 100)
P_range= np.linspace(1.35, 13.5, 100)
T= 353.15
T_C= T-273.15
n_d= 2.5 # EOD Coefficient
P_vessel_a= p_anode
T_ref = 303.0

# Hydrogen parameters
D_H2_ref = 1.0e-10
E_D_H2 = 20000
H_H2_ref = 7.0e4
dH_sol_H2 = 5000

# Oxygen parameters
D_O2_ref = 5.0e-11
E_D_O2 = 22000
H_O2_ref = 1.3e5
dH_sol_O2 = 6000

def D_H2(T):
    return D_H2_ref * np.exp(-E_D_H2/R * (1/T - 1/T_ref))

def D_O2(T):
    return D_O2_ref * np.exp(-E_D_O2/R * (1/T - 1/T_ref))

def H_H2(T):
    return H_H2_ref * np.exp(dH_sol_H2/R * (1/T - 1/T_ref))

def H_O2(T):
    return H_O2_ref * np.exp(dH_sol_O2/R * (1/T - 1/T_ref))

def eps_H2(T):
    return D_H2(T) / H_H2(T)

def eps_O2(T):
    return D_O2(T) / H_O2(T)

# ---- Mass transfer coefficient ----
D_H2_liq = 5e-9
delta_mt = 50e-6

k_mt = D_H2_liq * (eps / tau) / delta_mt

results = {
    "Stack Power": [], "Faradaic Efficiency": [], "Stack Current": [], "Stack Voltage": [], "H2_in_O2_percent": [], "O2_in_H2_percent": [], "Physical Faradaic Efficiency": [], "Current Density": [],
    "H2 Yield": [], "O2 Yield": [], "H2O Yield": [], "H2O Inlet Mass": [], "H2O Diffusion": [], "H2O Electro-osmotic Drag": [], "H2O Hydraulic Pressure": [], "H2O Membrane Total": [], "H2O Outlet Anode": [],
    "H2O Outlet Cathode": [], "H2O Consumed": [], "Process to Cooling Water Ratio": [], "SEC": []
}
for P_goal in P_range:
        # Define a small helper function to find where (Calculated P - Goal P) == 0
        def power_error(j_guess):
            V = V_cell(T, j_guess, p_cathode, p_anode)
            I = j_guess * A_cell
            P_calc = (V * N_cells * I) / 1000
            return P_calc - P_goal

        # Solve for the j that gives us P_goal (starting guess of 0.5)
        j_solved = fsolve(power_error, 0.5)[0]
        I_stack= j_solved*A_cell          # A     Stack current 
        V_stack= V_cell(T, j_solved, p_cathode, p_anode)*N_cells # V     Stack voltage
        P_stack= V_stack*I_stack/1000 # kW    Stack power
        P_bop_kw = 0.04 * 13.5
        n_dot_h2_theo = (I_stack * N_cells) / (z * F)
        n_dot_o2_theo = (I_stack * N_cells) / (2 * z * F)
        # --- Gas Crossover Part ---
        # ---- New H2 crossover model ----

        Dmem_H2 = D_H2(T)
        Hmem_H2 = H_H2(T)

        #C_H2_gen = n_dot_h2_gen / (k_mt * A_m2 * N_cells)

        # ---- Two-phase PEMWE crossover closure ----

        # Henry equilibrium (dissolved limit)
        C_eq = (p_cathode * 1e5) / Hmem_H2   # mol/m^3

        # Mass transfer capacity (removal rate of dissolved H2)
        J_mt = k_mt * A_m2 * N_cells         # mol/s per concentration driving force

        # Production-driven "supersaturation tendency"
        C_gen = n_dot_h2_theo / (J_mt + 1e-30)  # mol/m^3

        # --- Hydrogen crossover: Henry's-law-based physical model ---
        # 1. Determine the concentration of hydrogen dissolved at the membrane boundary (Henry's Law)
        # It depends directly on cathode pressure, meaning it stays active even at low loads!
        C_H2_mem = (p_cathode * 1e5) / Hmem_H2  # mol/m^3

        # 2. Calculate the physical molar crossover leakage rate across the membrane thickness
        n_perm_h2_stack = (A_m2 * N_cells * Dmem_H2 / L_pem_m) * C_H2_mem  # mol/s

        # 3. Compute true physical Faradaic Efficiency
        # As n_dot_h2_theo gets small at low loads, this fraction drops, producing the characteristic U-shaped efficiency curve
        eta_F_physical = (n_dot_h2_theo - n_perm_h2_stack) / n_dot_h2_theo
        n_dot_h2_gen = n_dot_h2_theo * eta_F_physical
        # Corrections
        eta_F= a_corrected*(j_solved)**b + c
        n_dot_o2_gen = (I_stack * N_cells) / (2*z* F)* eta_F_physical
        n_dot_o2_outlet= n_dot_o2_gen
        n_dot_h2o_consumed = eta_F_physical * (I_stack* N_cells / (z * F))
        h2_in_o2_percent = (n_perm_h2_stack / (n_dot_o2_gen + n_perm_h2_stack)) * 100
        # ---- Oxygen crossover model ----
        Dmem_O2 = D_O2(T)
        Hmem_O2 = H_O2(T)

        # Mass transfer capacity for Oxygen (can be different from H2, but often assumed similar)
        J_mt_O2 = k_mt * A_m2 * N_cells         

        # Production-driven supersaturation at the Anode
        # n_dot_o2_gen is the calculated O2 production rate
        C_gen_O2 = n_dot_o2_gen / (J_mt_O2 + 1e-30) 

        # Henry equilibrium (Anode side is usually at lower pressure, e.g., 1 bar)
        C_eq_O2 = (p_anode * 1e5) / Hmem_O2

        # Local concentration of O2 at the membrane interface
        C_O2_mem = C_eq_O2 * (1 + C_gen_O2 / (C_eq_O2 + C_gen_O2))

        # Molar crossover flux of Oxygen to the Cathode
        n_perm_o2_stack = (A_m2 * N_cells * Dmem_O2 / L_pem_m) * C_O2_mem

        # Concentration of O2 in the Hydrogen product stream (%)
        o2_in_h2_percent = (n_perm_o2_stack / (n_dot_h2_gen + n_perm_o2_stack)) * 100

        n_h2o_inlet_anode= 20*n_dot_h2o_consumed
        h2o_inlet_mass= n_h2o_inlet_anode* M_H2O
        # Membrane part
        n_dot_h2o_eod= n_d*I_stack* N_cells/F # Electro osmotic drag
        
        # Diffusion part
        log_Psat = 5.40221 - (1838.675 / (T - 31.737))
        P_sat = (10**log_Psat) * 1e5  # Convert bar to Pa
        rho_cat_vap = (P_sat * M_H2O) / (R * T)  # kg/m^3- H2O vapour density
        C_an = water_density(T_C) / M_H2O
        C_cat = rho_cat_vap / M_H2O
        # 2) Number of H2O species (Molar Fluxes)
        n_an_h2o= n_dot_h2o_eod+ n_dot_h2o_consumed
        n_cat_h2o= n_dot_h2o_eod
        # 3) Diffusion Coefficient of Water in Membrane (Dw)
        Dw = 5.5e-11 * np.exp(2416 * (1/303 - 1/T))

        # 4) Effective Binary Diffusion Coefficients (Fuller Method)
        # Reference values at 298K, 1 atm (1.01325e5 Pa)
        # H2O-O2 (Anode) and H2O-H2 (Cathode)
        D12_an_ref = 2.5e-5 
        D12_cat_ref = 8.0e-5

        D12_an = D12_an_ref * ((T/298)**1.75) * (1.01325e5 / p_anode*1e5)
        D12_cat = D12_cat_ref * ((T/298)**1.75) * (1.01325e5 / p_cathode*1e5)
        Dan_eff = D12_an * (eps / tau)
        Dcat_eff = D12_cat * (eps / tau)
        n_diff_term1_cathode= C_cat+ (t_cathode*n_cat_h2o/Dcat_eff)
        n_diff_term2_anode= C_an- (t_anode*n_an_h2o/Dan_eff)
        n_dot_h2o_diff= A_m2*Dw*(n_diff_term1_cathode- n_diff_term2_anode)/(L_pem_m)
        n_dot_h2o_diff_stack= n_dot_h2o_diff*N_cells
        
        # Pressure gradient part
        pressure_diff= (p_cathode- p_anode)*1e5
        n_p_cell = A_m2 * (K_eff * water_density(T_C) / (dynamic_viscosity(T) * M_H2O)) * (pressure_diff / L_pem_m)
        n_pe_stack= n_p_cell*N_cells
        n_dot_h2o_membrane= abs(n_dot_h2o_diff_stack)+ n_dot_h2o_eod- n_pe_stack
        n_dot_h2o_outlet_anode= n_h2o_inlet_anode- n_dot_h2o_membrane- n_dot_h2o_consumed
        
        # Cathode Balance
        n_h2o_outlet_cathode= n_dot_h2o_membrane
        n_dot_h2_outlet_cathode= n_dot_h2_gen
        pw_to_cw_ratio= h2o_inlet_mass*100/mass_rate
        # sec= (P_stack+0.5)/(n_dot_h2_gen*0.002016*3600)
        sec= (P_stack)/(n_dot_h2_gen*M_H2*3600)
        # max_h2_in_o2_percent = max(h2_in_o2_percent)
        results["Stack Power"].append(P_stack)
        results["Faradaic Efficiency"].append(eta_F)
        results["Stack Current"].append(I_stack)
        results["Stack Voltage"].append(V_stack)
        results["H2_in_O2_percent"].append(h2_in_o2_percent)
        results["O2_in_H2_percent"].append(o2_in_h2_percent)
        results["Physical Faradaic Efficiency"].append(eta_F_physical)
        results["H2 Yield"].append(n_dot_h2_gen)
        results["O2 Yield"].append(n_dot_o2_gen)    
        results["H2O Yield"].append(n_dot_h2o_consumed)
        results["H2O Inlet Mass"].append(h2o_inlet_mass)
        results["H2O Diffusion"].append(n_dot_h2o_diff_stack)
        results["H2O Electro-osmotic Drag"].append(n_dot_h2o_eod)
        results["H2O Hydraulic Pressure"].append(n_pe_stack)
        results["Current Density"].append(j_solved)
        results["H2O Membrane Total"].append(n_dot_h2o_membrane)
        results["H2O Outlet Anode"].append(n_dot_h2o_outlet_anode)
        results["H2O Outlet Cathode"].append(n_h2o_outlet_cathode)    
        results["H2O Consumed"].append(n_dot_h2o_consumed)
        results["Process to Cooling Water Ratio"].append(pw_to_cw_ratio)
        results["SEC"].append(sec)
                   
        

if __name__ == "__main__":
    plt.figure(figsize=(12, 6))
    plt.plot(results["Stack Power"],results["H2_in_O2_percent"], linestyle='-', color='b')
    plt.axhline(2, linestyle='--', color='r', label='2%')
    plt.axvline(1.35, linestyle='--', color='g', label='Minimum Stack Power')
    plt.xlabel("Stack Power (kW)")
    plt.ylabel("H2 in O2 (%)")
    plt.title("H2 in O2 % vs Stack Power at {} °C".format(T_C), fontsize=14)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(results["Stack Power"],results["O2_in_H2_percent"], linestyle='-', color='b')
    plt.axvline(1.35, linestyle='--', color='g', label='Minimum Stack Power')
    plt.xlabel("Stack Power (kW)")
    plt.ylabel("O2 in H2 (%)")
    plt.title("O2 in H2 % vs Stack Power at {} °C".format(T_C), fontsize=14)
    plt.legend()        
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(results["Stack Power"],results["Physical Faradaic Efficiency"], label='Physical Faradaic Efficiency', linestyle='-', color='b')
    plt.plot(results["Stack Power"],results["Faradaic Efficiency"], label='Faradaic Efficiency', linestyle='-', color='r')
    plt.axvline(1.35, linestyle='--', color='g', label='Minimum Stack Power')
    plt.xlabel("Stack Power (kW)")
    plt.ylabel("Faradaic Efficiency")
    plt.title(" Faradaic Efficiency vs Stack Power at {} °C".format(T_C), fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(results["Current Density"], results["Physical Faradaic Efficiency"], label='Physical Faradaic Efficiency', linestyle='-', color='b')
    plt.plot(results["Current Density"], results["Faradaic Efficiency"], label='Faradaic Efficiency', linestyle='-', color='r')
    plt.xlabel("Stack Current Density (A/cm^2)")
    plt.ylabel("Efficiencies with Current (%)")
    plt.title("Faradaic Efficiency vs Stack Current at {} °C".format(T_C), fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Plot 5: Water Transport Mechanisms ---
    plt.figure(figsize=(12, 6))
    plt.plot(results["Current Density"], results["H2O Electro-osmotic Drag"], label='Electro-osmotic Drag', color='blue')
    plt.plot(results["Current Density"], results["H2O Diffusion"], label='Back Diffusion', color='green')
    plt.plot(results["Current Density"], results["H2O Hydraulic Pressure"], label='Hydraulic Pressure', color='orange')
    plt.xlabel("Current Density (A/cm^2)")
    plt.ylabel("Molar Flow Rate (mol/s)")
    plt.title("Water Transport Mechanisms across Membrane at {} °C".format(T_C))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate the water transport ratios from the results
    membrane_ratio = [results["H2O Membrane Total"][i] / results["H2O Consumed"][i] for i in range(len(results["Current Density"]))]
    anode_outlet_ratio = [results["H2O Outlet Anode"][i] / results["H2O Consumed"][i] for i in range(len(results["Current Density"]))]

    plt.figure(figsize=(12, 6))

    # Plot Ratio of Membrane Transport vs Consumption
    plt.plot(results["Current Density"], membrane_ratio, color='teal', label='Membrane Transport / Water Consumed', linewidth=2)

    # Plot Ratio of Anode Outlet vs Consumption
    plt.plot(results["Current Density"], anode_outlet_ratio, color='crimson', label='Anode Outlet / Water Consumed', linestyle='--')

    plt.xlabel("Current Density (A/cm^2)")
    plt.ylabel("Ratio (Multiple of Consumption)")
    plt.title("Relative Water Flow Ratios at {} °C".format(T_C))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(results["Current Density"], results["Process to Cooling Water Ratio"], label='Ratio of Process water to Cooling Water(%)', linestyle='-', color='b')
    plt.xlabel("Current Density (A/cm^2)")
    plt.ylabel("Ratio of Process water to Cooling Water(%)")
    plt.title("Ratio of Process water to Cooling Water vs Current Density at {} °C".format(T_C), fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(results["Stack Power"], results["SEC"], label='SEC', linestyle='-', color='b')
    plt.xlabel("Stack Power (kW)")
    plt.ylabel("SEC")
    plt.title("SEC vs Power at {} °C".format(T_C), fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    
    print(f"Stack Current: {I_stack:.4f} A, Power: {P_stack:.4f} kW, Stack Voltage: {V_stack:.4f} V, Faradaic Efficiency: {eta_F_physical*100:.4f} %")
    print(f"H2 Yield: {n_dot_h2_gen:.4f} mol/s, O2 Yield: {n_dot_o2_gen:.4f} mol/s, H2O Yield: {n_dot_h2o_consumed:.4f} mol/s")
    print(f"H2O Inlet Mass: {h2o_inlet_mass:.4f} kg/s which is {h2o_inlet_mass*100/mass_rate:.4f} % relative to the cooling water mass rate")
    print(f"H2O Diffusion: {n_dot_h2o_diff_stack:.4f} mol/s")
    print(f"H2O electro osmotic drag: {n_dot_h2o_eod:.4f} mol/s")
    print(f"H2O hydraulic pressure gradient: {n_pe_stack:.4f} mol/s")
    print(f"H2O Membrane Total: {n_dot_h2o_membrane:.4f} mol/s which is {n_dot_h2o_membrane/n_dot_h2o_consumed:.4f} times relative to the water consumed at the anode")
    print(f"H2O Outlet: {n_dot_h2o_outlet_anode:.4f} mol/s which is {n_dot_h2o_outlet_anode/n_dot_h2o_consumed:.4f} times relative to the water consumed at the anode")
    print(f"Hydrogen in Oxygen Stream: {h2_in_o2_percent:.4f} %")
    print(f"Maximum Current Density: {j_solved:.4f} A/cm^2")
    print(f"Hydrogen yield in kg/s:{n_dot_h2_gen*0.002016:.4f} kg/s")
    print(f" SEC: {sec:.4f} KWh/kg")
