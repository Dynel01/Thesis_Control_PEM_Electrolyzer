# ── Cell Parameters (Olivier et al. 2017, Table 3) ────────────────────────
A_cell    = 130.0         # cm²   MEA active area
N_cells   = 48            # -     cells per stack
j_nom     = 0.92          # A/cm² nominal current density
j_max= 1.2074              # A/cm² maximum current density

# ── Operating Conditions ───────────────────────────────────────────────────
p_cathode = 13.0          # bar   hydrogen side (target delivery pressure)
p_anode   = 1.01325       # bar   oxygen side (atmospheric)
T_init    = 20 + 273.15   # K     cold start temperature (ambient)
T_min     = 40 + 273.15   # K     minimum operating temperature
T_nom     = 55 + 273.15   # K     Olivier nominal (validation reference)
T_op      = 65 + 273.15   # K     target operating temperature
T_max     = 80 + 273.15   # K     maximum before cooling activates

# ── Activation Parameters (Olivier et al. 2017, Table 7) ──────────────────
j0_anode_ref   = 2.00e-7  # A/cm²  anode ref exchange current density
j0_cathode_ref = 0.0447   # A/cm²  cathode ref exchange current density
Ea_anode       = 61100    # J/mol  anode activation energy
Ea_cathode     = 23800    # J/mol  cathode activation energy
T0_ref         = 298.15   # K      reference temperature
lambda_m = 14.0    # -   fully hydrated membrane (Springer et al. 1991)
M_H2 = 2.016e-3           # kg/mol
M_H2O = 18.015e-3         # kg/mol
M_O2 = 32.00e-3           # kg/mol



# Stack Dimensions
t_CC= 1e-3 # m
t_BP= 2e-3 # m
from Resistance_Calc import L_pem_cm, R_others
import numpy as np
L_pem_m = L_pem_cm*1e-2 # m
t_EP= 2e-2 # m
n_CC= 96
n_BP= 47
n_EP= 2
n_MEA= 48

# Stack Geometry
L_CC= n_CC*t_CC
L_BP= n_BP*t_BP
L_EP= n_EP*t_EP
L_MEA= L_pem_m*n_MEA
L_Stack= L_CC+L_BP+L_EP+L_MEA
Stack_width, Stack_breadth = np.sqrt(A_cell), np.sqrt(A_cell)
Plate_width, Plate_breadth = 0.18, 0.18

# Channel Geometry
N_ch= 48
N_stack_channels= N_ch*n_BP
ch_width= 1.5e-3 # m
ch_breadth= 1e-3 # m
ch_area= ch_width*ch_breadth
ch_perimeter= 2*(ch_width+ch_breadth)
ch_hydraulic_diameter= 4*ch_area/ch_perimeter

# Specific Heat Capacities [J/kg·K]
CP_TITANIUM = 523.0    # Bipolar plates and PTL
CP_STEEL = 500.0       # End plates and bolts
CP_WATER = 4180.0      # Internal cooling/process water
CP_MEMBRANE = 1200.0   # Nafion / Polymer MEA

# -- Physical Densities [kg/m3] --
RHO_TITANIUM = 4510.0
RHO_STEEL    = 8000.0
RHO_WATER    = 1000.0
RHO_NAFION   = 2000.0

# -- Olivier et al. Ratios --
BP_SOLID_RATIO = 0.625    # 62.5% Titanium, 37.5% water channels
PTL_POROSITY   = 0.35     # 35% of PTL volume is water

# Convert Area to m2 for calculations
A_m2 = A_cell / 10000.0 
Plate_area = Plate_width * Plate_breadth
# Bipolar Plates (47 plates, 2mm thick)
V_BP_total = n_BP * Plate_area * t_BP
M_BIPOLAR_PLATES = V_BP_total * BP_SOLID_RATIO * RHO_TITANIUM  # ~3.44 kg


# PTLs (96 layers, 1mm thick)
V_PTL_total = n_CC * A_m2 * t_CC
M_PTL = V_PTL_total * (1 - PTL_POROSITY) * RHO_TITANIUM        # ~3.54 kg


# End Plates (2 plates, 20mm thick)
V_EP_total = n_EP * Plate_area * t_EP
M_END_PLATES = V_EP_total * RHO_STEEL                          # ~4.16 kg

# MEA (48 layers, 125 micron)
V_MEA_total = n_MEA * A_m2 * L_pem_m
M_MEA = V_MEA_total * RHO_NAFION                               # ~0.16 kg

# Internal Water (Filling the BP channels and PTL pores)
V_water_internal = (V_BP_total * (1 - BP_SOLID_RATIO)) + (V_PTL_total * PTL_POROSITY)
M_INTERNAL_WATER = V_water_internal * RHO_WATER                # ~0.92 kg

# Calculate individual capacitances [J/K]
C_BP    = M_BIPOLAR_PLATES * CP_TITANIUM
C_PTL   = M_PTL            * CP_TITANIUM  # Corrected to Titanium cp
C_EP    = M_END_PLATES     * CP_STEEL
C_WATER = M_INTERNAL_WATER * CP_WATER
C_MEA   = M_MEA            * CP_MEMBRANE

# Total System Capacitance
C_TOTAL_J_K = C_BP + C_PTL + C_EP + C_WATER + C_MEA
C_TOTAL_KJ_K = C_TOTAL_J_K / 1000.0  # Used directly in the thermal simulation (~18.36)

# Process Water Channels
p_ch_width= 1.5e-3 # m
p_ch_breadth= 0.5e-3 # m
p_ch_area= p_ch_width*p_ch_breadth
p_ch_volume= p_ch_area*Stack_width*1e-2*N_stack_channels
p_an_volume= p_ch_volume

t_cathode= (L_BP/2)+ t_CC # Cathode thickness
t_anode= t_cathode  # Same thickness as cathode

# PTL Properties
eps = 0.35          # PTL Porosity
tau = eps**-0.5     # Tortuosity (Bruggeman)

# 1) Permeability of Nafion (Standard literature value)
K_darc = 1.58e-19   # [m^2]
K_eff= K_darc * (eps / tau) # Effective Permeability

if __name__ == "__main__":
    print(f"Effective Permeability: {K_eff:.4f} m2")
    L_Stack= L_CC+L_BP+L_EP+L_MEA
    print(f"Stack Length: {L_Stack:.4f} m")
    print(f"Stack Width: {Stack_width:.4f} cm")
    print(f"Stack Breadth: {Stack_breadth:.4f} cm")
    print(f"Number of Channels: {N_stack_channels}")
    print(f"Hydraulic Diameter: {ch_hydraulic_diameter:.4f} m")
    print(f"Bipolar Plates Mass: {M_BIPOLAR_PLATES:.4f} kg")
    print(f"PTL Mass: {M_PTL:.4f} kg")
    print(f"End Plates Mass: {M_END_PLATES:.4f} kg")
    print(f"Membrane Mass: {M_MEA:.4f} kg")
    print(f"Internal Water Mass: {M_INTERNAL_WATER:.4f} kg")
    print(f"Total Stack Mass: {M_BIPOLAR_PLATES + M_PTL + M_END_PLATES + M_INTERNAL_WATER + M_MEA:.4f} kg")
    print(f"Total Thermal Capacity: {C_TOTAL_J_K:.4f} J/K")
    print(f"BP Thermal Capacity: {C_BP:.4f} J/K")
    print(f"PTL Thermal Capacity: {C_PTL:.4f} J/K")
    print(f"End Plates Thermal Capacity: {C_EP:.4f} J/K")
    print(f"Water Thermal Capacity: {C_WATER:.4f} J/K")
    print(f"MEA Thermal Capacity: {C_MEA:.4f} J/K")
    print(f"Process Water Channels Area: {p_ch_area:.4f} m2")
    print(f"Process Water Channels Volume: {p_ch_volume:.4f} m3 or {p_ch_volume*1000:.4f} L")