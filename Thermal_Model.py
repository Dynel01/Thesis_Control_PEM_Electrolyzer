from PEMWE_Parameters import *
from Electrochemical_Model import *
import numpy as np
import matplotlib.pyplot as plt
# SINGLE 13.5 kW stack unit
N_cells = 48              # Cells in that one stack
N_stacks_total = 10000    # Total units in your 210 MW Hub
T_nominal = 328.15       # 55°C (Fixed Operating Point)
V_thermoneutral = V_tn(T_nominal)   # V (Thermoneutral voltage - Majumdar Eq 14)
T_ambient = 298.15       # 25°C (Ambient Temperature)
V_temp= V_cell(T_nominal,j_nom, p_cathode=13, p_anode=1.01325)
I_stack= j_nom*130
v_air= 0.2 # m/s
mu_air= 1.56e-5 # m^2/s (Kinematic viscosity of air)
U= 1975 # W/(m^2·K) (Overall Heat Transfer Coefficient)

Reynolds_number = v_air*L_Stack/mu_air
Prandtl_number = 0.7
Nusselt_number = 0.66*Reynolds_number**0.675*Prandtl_number**(1/3)
k_air= 0.026 # W/(m·K)
h_air_conv= Nusselt_number*k_air/L_Stack  # W/(m^2·K)
Perimeter= 2*(Plate_width+Plate_breadth)
LSA= Perimeter*L_Stack # Lateral Surface Area (m^2)
Emissivity= 0.6
S_B_constant= 5.68e-8 # W/(m^2·K^4)
h_air_rad= 4*Emissivity*S_B_constant*((T_nominal+T_ambient)/2)**3
h_total= h_air_conv + h_air_rad
R_th= 1/h_total
t_ins= 0.02 # m
k_ins= 0.045 # W/(m·K)
R_ins= t_ins/k_ins
R_th_total= R_th + R_ins
h_total_ins= 1/R_th_total
q_vap_stack= 0 # J
T_celsius= T_nominal-273.15
T_inlet_cw = 288.15    # [K] 35°C Cooling water supply temperature
T_outlet_cw = T_inlet_cw+5   # [K] 40°C Cooling water outlet temperature
T_water_avg = T_inlet_cw + (T_outlet_cw - T_inlet_cw) / 2  # [K] Average temperature of cooling water
def water_density(T_celsius):
    rho = (999.83952 + 16.945176 * T_celsius - 
           7.9870401e-3 * T_celsius**2 - 
           46.170461e-6 * T_celsius**3 + 
           105.56302e-9 * T_celsius**4 - 
           280.54253e-12 * T_celsius**5) / (1 + 16.897850e-3 * T_celsius)
    return rho
def dynamic_viscosity(T_nominal):   
    """
    Dynamic Viscosity [Pa.s]
    T_nominal : temperature [K]
    """
    return 2.414e-5 * 10**(247.8/(T_nominal-140))

def cp_water(T_nominal):   
    """
    Specific Heat [J/kg.K]
    T_nominal : temperature [K]
    """
    A, B, C, D, E = 12010.1471, -80.4072879, 0.309866489, -0.000538186884, 0.00000036223854
    cp = A + B*T_nominal + C*T_nominal**2 + D*T_nominal**3 + E*T_nominal**4
    return cp


def thermal_balance(V_temp, I_stack, T_nominal, N_stacks_total, N_cells, LSA, h_total_ins, T_ambient, V_thermoneutral):
    '''
    Calculates the Heat generated and cooling required to maintain the required Temperature
    '''
    q_gen_stack= N_cells * I_stack * (V_temp - V_thermoneutral)
    q_gen_cell= q_gen_stack/ N_cells
    q_density= q_gen_cell/(A_cell*1e-4) # W/m^2
    q_loss_stack= LSA *h_total_ins * (T_nominal - T_ambient)
    q_cooling_required_stack = max(0, q_gen_stack - q_loss_stack-q_vap_stack)
    # q_cooling_required_stack = 0.0
    Temp_diff= T_nominal - T_water_avg
    q_limit_physical= U*Plate_area*Temp_diff
    # 4. Scale to 210 MW Hub (Convert to Megawatts)
    total_cooling_MW = (q_cooling_required_stack * N_stacks_total) / 1e6
    return total_cooling_MW, q_cooling_required_stack, q_gen_stack, q_loss_stack, q_density, q_limit_physical

total_cool_MW, q_cooling_stack, q_generated_stack, q_losses_stack, q_density, q_limit_physical = thermal_balance(V_temp, I_stack, T_nominal, N_stacks_total, N_cells, LSA, h_total_ins, T_ambient, V_thermoneutral)

mass_rate= q_cooling_stack/(cp_water(T_nominal)*5)
channel_mass_rate= mass_rate/N_stack_channels
Mass_velocity= channel_mass_rate/ch_area
flow_velocity= channel_mass_rate/(ch_area*water_density(T_celsius))
def Reynolds_number_water(Mass_velocity, dynamic_viscosity, ch_hydraulic_diameter):
    return Mass_velocity*ch_hydraulic_diameter/dynamic_viscosity

Re= Reynolds_number_water(Mass_velocity, dynamic_viscosity(T_nominal), ch_hydraulic_diameter)

# Temperature Gradients
dT_dx= 5/Plate_width # K/m (Inlet and Outlet Temperature)
dy_T= q_density/ U
dT_dy=dy_T/0.004 # K/m

# Pressure Drop
f= 60/Re # Darcy Friction Factor
water_density_now= water_density(T_celsius)
def Pressure_drop(f,L_stack,ch_hydraulic_diameter, water_density_now, flow_velocity):
    '''
    f: Darcy Friction Factor
    L_stack: Length of the stack [m]
    ch_hydraulic_diameter: Hydraulic Diameter of the channel [m]
    water_density_now: Density of water at current temperature [kg/m^3]
    flow_velocity: Flow Velocity [m/s]
    '''
    return f*(L_stack/ch_hydraulic_diameter)*(water_density_now * (flow_velocity**2) / 2)
P_drop= Pressure_drop(f,L_Stack,ch_hydraulic_diameter, water_density_now, flow_velocity)

# Warm Start Time
time_required= (C_TOTAL_J_K*(T_nominal - T_ambient))/q_generated_stack-q_losses_stack
print(f"Time: {time_required:.2f} seconds or {time_required/60:.2f} minutes")

if __name__=="__main__":
    print(f"I_stack: {I_stack:.4f} A, V_cell: {V_temp:.4f} V, V_tn: {V_thermoneutral:.4f} V")
    print(f"Reynolds Number: {Reynolds_number:.4f}")
    print(f"Nusselt Number: {Nusselt_number:.4f}")
    print(f" LSA: {LSA:.4f} m^2")
    print(f" h_air_rad: {h_air_rad:.4f} W/(m^2·K)")
    print(f"Dynamic Viscosity: {dynamic_viscosity(T_nominal):.4f} Pa.s")
    print(f"Specific Heat: {cp_water(T_nominal):.4f} J/kg.K")
    print(f'q_gen_stack: {q_generated_stack:.4f} W')
    print(f'q_loss_stack: {q_losses_stack:.4f} W')
    print(f'q_cooling_required_stack: {q_cooling_stack:.4f} W')
    print(f'Total Cooling Required in Megawatts: {total_cool_MW:.4f}')
    print(f"Mass Rate: {mass_rate:.4f} kg/s")
    print(f"Channel Mass Rate: {channel_mass_rate:.4f} kg/s")
    print(f"Mass Velocity: {Mass_velocity:.4f} kg/m^2.s")
    print(f"Flow Velocity: {flow_velocity:.4f} m/s")
    print(f"Cooling Limit: {q_limit_physical:.4f} W")
    print(f"Reynolds Number of Cooling Channel: {Re:.4f}")
    print(f"dT/dx: {dT_dx:.4f} K/m")
    print(f"Temperature Difference: {dy_T:.4f} K")
    print(f"dT/dy: {dT_dy:.4f} K/m")
    print(f"Pressure Drop: {P_drop:.4f} Pa")
        
    




