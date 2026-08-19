import PEMWE_Parameters as PEM
import Electrochemical_Model as ECM
import Thermal_Model as TM
import Mass_balance_model as MM
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

stack_power=np.linspace(0,135e3,21)
# SOH_range=np.linspace(1,0.1,10)
SOH_range=np.linspace(1,1,1) #1.0
temperatures = np.linspace(25, 80, 12) + 273.15
N_stacks_total=1e4
dt_seconds= 600
stack_capacity_kw= 13.5
N_stacks_total = 10000
stack_capacity_kw = 13.5
capex_per_kw = 494
CAPEX_total = capex_per_kw * stack_capacity_kw * N_stacks_total
                # Dynamic update of membrane thickness based on active SOH

                # Nested resolution loop: temperature -> SOH -> power setpoint
results_table=[]
for current_T in temperatures:
    for current_SOH in SOH_range: 
        current_L_pem_m = PEM.L_pem_m * (0.5 + 0.5 * current_SOH) 
        cumulative_degradation_V=0+ 0.18*(1-current_SOH) 
        V_thermoneutral = ECM.V_tn(current_T)             
        for p_val in stack_power:
            if p_val == 0:
                I_stack = 0
                V_cells = 0
                V_stack = 0
                P_stack_actual = 0
                h2_gen = 0
                power_ratio = 0
                j_solved= 0
                h2_in_o2_percent= 0
                h2_gen= 0
                degradation_cost= 0
                
            else:
                stack_power_goal_kw = p_val/1e4
                def power_error(j_guess):
                    V_ideal = ECM.V_cell(current_T, j_guess, PEM.p_cathode, PEM.p_anode)
                    V_degraded = V_ideal + cumulative_degradation_V
                    I = j_guess * PEM.A_cell
                    P_calc = (V_degraded * PEM.N_cells * I) / 1000.0
                    return P_calc - stack_power_goal_kw

                # Solve for the current density that matches the target power setpoint
                j_solved = fsolve(power_error, 0.5)[0]
                I_stack = j_solved * PEM.A_cell
                V_ideal_step = ECM.V_cell(current_T, j_solved, PEM.p_cathode, PEM.p_anode)
                V_cells = V_ideal_step + cumulative_degradation_V
                V_stack = V_cells * PEM.N_cells
                P_stack_actual = (V_stack * I_stack) / 1000.0

                            # ── PHASE 2: THERMAL MODULE ──
                            # Calculate the thermoneutral voltage for heat balance equations
                V_thermoneutral = ECM.V_tn(current_T)
                            
                            # Track reference thermoneutral voltage even when idle
                current_L_pem_m = PEM.L_pem_m * (0.5 + 0.5 * current_SOH)
                                
                q_gen_stack= PEM.N_cells * I_stack * (V_cells - V_thermoneutral)
                q_loss_stack= TM.LSA*TM.h_total_ins*(current_T-TM.T_ambient)
                            # Temp_diff= self.target_T - TM.T_water_avg
                Temp_diff= 10
                q_cooling_stack= q_gen_stack+q_loss_stack
                q_limit_physical= TM.U*TM.Plate_area*Temp_diff*PEM.N_cells
    # 3. Calculate the electricity cost to heat the water
    # We only pay to heat what we need to get to 70C
                n_dot_h2_theo = (I_stack * PEM.N_cells) / (MM.z * MM.F)
                n_dot_o2_theo = (I_stack * PEM.N_cells) / (2 * MM.z * MM.F)
                    
                    # Membrane cross-permeation physics parameters
                Dmem_H2 = MM.D_H2(current_T)
                Hmem_H2 = MM.H_H2(current_T)  
                C_H2_mem = (PEM.p_cathode * 1e5) / Hmem_H2
                n_perm_h2_stack = (PEM.A_m2 * PEM.N_cells * Dmem_H2 / current_L_pem_m) * C_H2_mem
                if n_dot_h2_theo > 0:
                    eta_F_physical = (n_dot_h2_theo - n_perm_h2_stack) / n_dot_h2_theo
                else:
                    eta_F_physical = 1.0
                                
                            # Calculate net gas output rates accounting for cross-permeation losses
                n_dot_h2_gen = max(0.0, n_dot_h2_theo * eta_F_physical)
                n_dot_o2_gen = max(0.0, n_dot_o2_theo * eta_F_physical)
                            
                        # Evaluate gas purity metrics (Hydrogen-in-Oxygen cross-over safety percentage)
                if (n_dot_o2_gen + n_perm_h2_stack) > 0:
                    h2_in_o2_percent = (n_perm_h2_stack / (n_dot_o2_gen + n_perm_h2_stack)) * 100.0
                else:
                    h2_in_o2_percent = 0.0
                # if h2_in_o2_percent > 2.0:
                #     continue            
                        # Integrated total mass generated over this 10-minute (600 seconds) discrete step (grams)
                h2_gen = n_dot_h2_gen* MM.M_H2*N_stacks_total      
                power_ratio = P_stack_actual / stack_capacity_kw
                if current_T< 40.0+273.15 or power_ratio < 0.1:
                    h2_gen = 0.0
                    h2_in_o2_percent = 0.0
                    q_cooling_stack = 0.0
                elif h2_gen > 0.0 and h2_in_o2_percent > 2.0: 
                    continue
                T_target_dh = 70.0+273.15  # District heating required temperature in °C
        
                T_cw_out= current_T- 7.5
                T_source = T_cw_out    # Electrolyzer operating temperature
                Q_source_total_MW = (q_cooling_stack * N_stacks_total) / 1e6
                # Minimum temperature difference for real heat exchangers (Pinch point guard)
                dT_pinch = 2.0 
                
                if T_source >= (T_target_dh - dT_pinch):
                    # Case A: Stack heat is hot enough to bypass the heat pump via an HX
                    W_pump_MW = 0.0
                    Q_delivered_MW = Q_source_total_MW  # Direct exchange (minus minor losses)
                else:
                    # Case B: Heat pump is required to lift the temperature
                    eta_hp = 0.45 
                    
                    # Carnot COP uses absolute temperatures (Kelvin)
                    COP_actual = eta_hp * (T_target_dh / (T_target_dh - T_source))
                    
                    # Fundamental Heat Pump Energy Balance (in MW)
                    W_pump_MW = Q_source_total_MW / (COP_actual - 1)
                    Q_delivered_MW = Q_source_total_MW + W_pump_MW
                    Q_net= Q_delivered_MW- W_pump_MW
                T_nominal_K = 60.0 + 273.15
                B_constant = 0.04  # Activation parameter for polymer chemical decay
                K_thermal = np.exp(B_constant * (current_T - T_nominal_K))
                step_steady_deg_V = ((0.2499 * j_solved + 2.3545) / 1e6) * K_thermal*(1/6)
                damage_fraction = step_steady_deg_V / 0.18
                degradation_cost = CAPEX_total*damage_fraction
                if p_val == 0:
                    state= "OFF"
                elif power_ratio == 0.1 and current_T < 40.0+273.15:
                    state= "COLD START"
                elif power_ratio == 0.05 and current_T >= 40.0+273.15:
                    state= "STANDBY"
                elif power_ratio > 0.1 and current_T >= 40.0+273.15:
                    state= "OPERATIONAL"
                else:
                    state= "ERROR"
            results_table.append({
                'Power': p_val,
                'Power ratio': power_ratio,
                'J_solved': j_solved,
                'V_cells': V_cells,
                'V_stack': V_stack,
                'I_stack': I_stack,
                'current_T': current_T,
                'current_SOH': current_SOH,
                'L_pem_m': current_L_pem_m,
                'h2_in_o2_percent': h2_in_o2_percent,
                'h2_gen_rate': h2_gen,
                'V_tn': V_thermoneutral,
                'Degradation_cost': degradation_cost
                    # Heat and revenue metrics are computed downstream in the MPC scripts
            })

            print("Successful Iteration")
results_df = pd.DataFrame(results_table)
results_df.to_csv('results_table.csv', index=False)


# # 1. Filter for SOH = 1.0
# df_soh1 = results_df[results_df['current_SOH'] == 1.0]

# # 1. Filter for specific temperatures to keep the plot clean
# # Converting Kelvin to Celsius for the legend
# temp_list = [45 + 273.15, 60 + 273.15, 75 + 273.15]
# import matplotlib.cm as cm
# colors = cm.viridis(np.linspace(0, 1, len(temp_list)))
# df_plot = results_df[results_df['current_T'].isin(temp_list)]

# # 2. Setup plotting
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # 3. Plot Q_delivered and Q_net
# # Using different styles for Delivered (Total) vs Net (Recovered)
# for temp in temp_list:
#     subset = df_plot[df_plot['current_T'] == temp]
#     t_celsius = temp - 273.15
#     color= colors[temp_list.index(temp)]
#     # Solid line for Delivered
#     ax1.plot(subset['Power'], subset['q_delivered'], linestyle='-',color=color, 
#              label=f'Delivered ({t_celsius:.0f}°C)')
#     # Dashed line for Net
#     ax1.plot(subset['Power'], subset['q_net'], linestyle='--',color=color, alpha=0.5,
#              label=f'Net Generated ({t_celsius:.0f}°C)')

# ax1.set_xlabel('Stack Power (W)')
# ax1.set_ylabel('Heat Energy (MW)')
# ax1.set_title('District Heating Delivery: Delivered vs. Net Generated', fontsize=14)
# ax1.legend(loc='upper left', ncol=2)
# ax1.grid(True, alpha=0.3)
# plt.axvline(13.5e3, linestyle='--', color='g', label='Minimum Stack Power')
# plt.axvline(135e3, linestyle='--', color='r', label='Maximum Stack Power')
# plt.show()