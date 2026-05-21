import PEMWE_Parameters as PEM
import Electrochemical_Model as ECM
import Thermal_Model as TM
import Mass_balance_model as MM
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class ElectrolyzerSimulation:
    def __init__(self, filename, plant_capacity_mw=135.0):
        # 1. Load data and format timestamps as index for chronological alignment
        self.data = pd.read_csv(filename)
        self.data['TimeStamp'] = pd.to_datetime(self.data['TimeStamp'])
        self.data.set_index('TimeStamp', inplace=True)
        
        # 2. Plant specs
        self.plant_capacity_kw = plant_capacity_mw * 1e3
        self.N_stacks_total = 10000
        
        # 3. Initial Physical States (These change over time!)
        self.current_T = 25 + 273.15  # Starts at 25C in Kelvin
        self.target_T = 60 + 273.15
        
    def determine_state(self, p_input_kw):
        """
        State Machine: Evaluates stack-level power and temperature 
        to determine the operating mode of the electrolyzer.
        """
        # 13.5 kW nominal capacity per stack. 10% limit = 1.35 kW
        if p_input_kw < 0.1*self.plant_capacity_kw:
            if self.current_T >= (40.0 + 273.15):
                return "STANDBY"  # Keep it warm, but 0 hydrogen output
            else:
                return "OFF"      # Let it cool down completely, 0 hydrogen output
        else:
            if self.current_T < (40.0 + 273.15):
                return "COLD START"
                
        # HARD GUARD: Even if the wind gives you 13.5 kW, cap it at 1.35 kW (10%) to heat safely!
            else:
                return "OPERATIONAL"
        
    def run_simulation(self):
        """Loops sequentially through the year, passing power and state to physics models."""
        results_list = []
        
        # Define the absolute timeline loop for the year 2014
        start_time = datetime(2014, 1, 1, 0, 0)
        end_time = datetime(2014, 12, 31, 23, 00)
        step_size = timedelta(minutes=10)
        dt_seconds = step_size.seconds
        operational_minutes_step = 0.0
        cold_start_minutes_step = 0.0
        current_time = start_time
        
        print("Running physical simulation chain...")
        while current_time <= end_time:
            
            # Default values if data is missing
            wind_power = 0.0
            status = "Missing"
            
            ramp_time_seconds = 0.0
            # Check if timestamp exists in our CSV
            if current_time in self.data.index:
                wind_power = self.data.loc[current_time, 'wtc_ActPower_mean']
                status = "Success"
            
            max_ramp_delta_kw = 0.10 * self.plant_capacity_kw * dt_seconds    
            # --- STEP 1: Power Limitation (Your logic) ---
            if wind_power > self.plant_capacity_kw:
                stack_power = self.plant_capacity_kw
            elif wind_power < 0.1*self.plant_capacity_kw:
                stack_power = 0
            else:
                stack_power = wind_power
                
            # --- STEP 2: Electrochemical Model ---
            # Example call using your imported ECM module:
            # current_density = ECM.calculate_current(stack_power, self.current_T, PEM.parameters)
            
            # --- STEP 3: Thermal Model (Updates the state for the next loop!) ---
            # Example call:
            # self.current_T = TM.calculate_temperature(self.current_T, stack_power, step_size.seconds)
            
            # --- STEP 4: Mass Balance ---
            # h2_production = MM.calculate_mass(stack_power)
            state = self.determine_state(stack_power)
            if state=="STANDBY":
                cooling_system_active = False
                pass
            elif state=="OFF":
                cooling_system_active = False
                pass
            elif state=="OPERATIONAL":
                cooling_system_active = True if self.current_T> self.target_T else False
                pass
            elif state=="COLD START":
                cooling_system_active = False
                stack_power= 0.1*self.plant_capacity_kw
                pass
            
            if state == "OPERATIONAL" or state == "COLD START":
                # Assign your target power goal for the nested function solver
                stack_power_goal_kw = stack_power/1e4
                
                # YOUR EXACT NESTED RESOLUTION FUNCTION
                def power_error(j_guess):
                    V = ECM.V_cell(self.current_T, j_guess, PEM.p_cathode, PEM.p_anode)
                    I = j_guess * PEM.A_cell
                    P_calc = (V * PEM.N_cells * I) / 1000.0
                    return P_calc - stack_power_goal_kw

                # Solve for current density using your specified fsolve setup
                j_solved = fsolve(power_error, 0.5)[0]
                I_stack = j_solved * PEM.A_cell
                V_stack = ECM.V_cell(self.current_T, j_solved, PEM.p_cathode, PEM.p_anode) * PEM.N_cells
                V_cell= V_stack/PEM.N_cells
                P_stack_actual = (V_stack * I_stack) / 1000.0

                # ── PHASE 2: THERMAL MODULE ──
                # Calculate the thermoneutral voltage for heat balance equations
                V_thermoneutral = ECM.V_tn(self.current_T)
            
            # Place your upcoming thermal model equations here, for example:
            # self.current_T_kelvin = TM.solve_temperature(self.current_T_kelvin, V_stack, I_stack, V_thermoneutral)
                
            else:
                # AS REQUESTED: If state is NOT operational, V and I are exactly 0
                j_solved = 0.0
                I_stack = 0.0
                V_stack = 0.0
                V_cell = 0.0
                P_stack_actual = 0.0
                
                # Track reference thermoneutral voltage even when idle
                V_thermoneutral = ECM.V_tn(self.current_T)
                
            q_gen_stack= PEM.N_cells * I_stack * (V_cell - V_thermoneutral)
            q_loss_stack= TM.LSA*TM.h_total_ins*(self.current_T-TM.T_ambient)
            Temp_diff= self.target_T - TM.T_water_avg
            q_limit_physical= TM.U*TM.Plate_area*Temp_diff
            if state=="COLD START":
                cooling_system_active = False
                q_cooling_stack = 0.0
                COLD_START_TIME= PEM.C_TOTAL_J_K*(PEM.T_min-self.current_T)/(q_gen_stack-q_loss_stack)
                print(f"Cold Start Time: {COLD_START_TIME:.2f} seconds at time {current_time}")
                if 0 < COLD_START_TIME < 600.0:
                    cold_start_minutes_step = COLD_START_TIME / 60.0
                    operational_minutes_step = 10.0 - cold_start_minutes_step
                else:
                    # The entire 10 minutes was spent warming up
                    cold_start_minutes_step = 10.0
                    operational_minutes_step = 0.0
                
            elif state == "OFF":
                cooling_system_active = False
                q_cooling_stack = 0.0
                cold_start_minutes_step = 0.0
                operational_minutes_step = 0.0
                

            elif state == "STANDBY":
                cooling_system_active = False
                q_cooling_stack = 0.0
                cold_start_minutes_step = 0.0
                operational_minutes_step = 0.0

            elif state == "OPERATIONAL":
                cold_start_minutes_step = 0.0
                operational_minutes_step = 10.0
                # Changed the ceiling guard to allow the math to execute during an overshoot
                if PEM.T_min <= self.current_T <= (PEM.T_max): 
                    temperature_error = self.current_T - self.target_T

                    # 1. BELOW TARGET TEMPERATURE: Predictive Warmup Control
                    if temperature_error < -0.1:
                        q_net_natural = q_gen_stack - q_loss_stack
                        
                        if q_net_natural > 0:
                            # Calculate energy needed to reach the target operating line
                            energy_to_target = PEM.C_TOTAL_J_K * (self.target_T - self.current_T)
                            time_to_target = energy_to_target / q_net_natural
                            
                            if time_to_target < dt_seconds:
                                # FIX: Target is reached mid-step! Blend warmup with equilibrium cooling
                                t_heating = time_to_target
                                t_steady = dt_seconds - t_heating
                                
                                cooling_system_active = True
                                q_cooling_stack = ((0.0 * t_heating) + (q_gen_stack - q_loss_stack) * t_steady) / dt_seconds
                                print(f"[{current_time}] 🔥 WARMUP TRANSITION: Reached target line mid-step in {time_to_target:.1f}s. Switching to equilibrium.")
                            else:
                                # Safe to heat naturally for the full 10 minutes
                                cooling_system_active = False
                                q_cooling_stack = 0.0
                                print(f"[{current_time}] 🟢 HEATING NATURAL: T = {self.current_T:.2f} K (Below target {self.target_T} K). No cooling needed.")
                        else:
                            cooling_system_active = False
                            q_cooling_stack = 0.0
                            print(f"[{current_time}] 🟢 HEATING NATURAL: Core temperature is flat or dropping naturally.")

                    # 2. NEAR TARGET TEMPERATURE: Maintain Perfect Equilibrium
                    elif abs(temperature_error) <= 0.1:
                        cooling_system_active = True
                        q_cooling_stack = max(0.0, q_gen_stack - q_loss_stack)
                        print(f"[{current_time}] 🔷 EQUILIBRIUM: T = {self.current_T:.2f} K. Cooling matches generation ({q_cooling_stack:.2f} W).")

                    # 3. ABOVE TARGET TEMPERATURE: Dynamic Time-Weighted Recovery
                    else:
                        cooling_system_active = True
                        q_cooling_steady = q_gen_stack - q_loss_stack
                        q_cooling_max = q_limit_physical
                        
                        excess_energy_joules = PEM.C_TOTAL_J_K * temperature_error
                        net_removal_capacity = q_cooling_max - q_cooling_steady
                        
                        if net_removal_capacity > 0:
                            time_to_cool_seconds = excess_energy_joules / net_removal_capacity
                            
                            if time_to_cool_seconds < dt_seconds:
                                # SCENARIO 1: The system recovers mid-step!
                                t_max = time_to_cool_seconds
                                t_steady = dt_seconds - t_max
                                q_cooling_stack = ((q_cooling_max * t_max) + (q_cooling_steady * t_steady)) / dt_seconds
                                print(f"[{current_time}] ✨ SCENARIO 1 (Mid-Step Cool): Recovered to target in {time_to_cool_seconds:.1f} seconds. "
                                      f"Spent remaining {t_steady:.1f}s idling at steady state.")
                            else:
                                # SCENARIO 2: System cools at maximum capacity for the entire step
                                q_cooling_stack = q_cooling_max
                                projected_drop = (net_removal_capacity / PEM.C_TOTAL_J_K) * dt_seconds
                                print(f"[{current_time}] 🟡 SCENARIO 2 (Full-Step Cool): Cooling flat out at max capacity ({q_cooling_max:.2f} W). "
                                      f"Temp dropping safely by ~{projected_drop:.2f} K this step.")
                        else:
                            # SCENARIO 3: Heat generation exceeds heat exchanger capacity! Overheating is inevitable.
                            q_cooling_stack = q_cooling_max
                            unpreventable_heat_flux = q_cooling_steady - q_cooling_max
                            dT_unpreventable = (unpreventable_heat_flux / PEM.C_TOTAL_J_K) * dt_seconds
                            print(f"[{current_time}] ❌❌ SCENARIO 3 (OVERWHELMED): Wind generation ({q_gen_stack:.2f} W) "
                                  f"exceeds total cooling limit ({q_cooling_max:.2f} W) + losses ({q_loss_stack:.2f} W)! "
                                  f"Stack will physically heat up by +{dT_unpreventable:.2f} K this step.")
                            
                        # Final safety boundary checks
                        q_cooling_stack = min(q_cooling_stack, q_limit_physical)
                        q_cooling_stack = max(0.0, q_cooling_stack)
                else:
                    # Fallback condition if the system is completely out of its physical constraints
                    cooling_system_active = False
                    q_cooling_stack = 0.0
                    print(f"[{current_time}] ⚠️ CRITICAL LIMIT: T = {self.current_T:.2f} K has breached safety boundaries.")    
                # Calculate the maximum power ramp rate based on your nominal stack power
                # --- COMPREHENSIVE POWER RAMP RATE CONTROLLER ---
                nominal_stack_kw = 13.5
                max_ramp_rate_kw_s = 0.10 * nominal_stack_kw  # 1.35 kW/s
                
                # Calculate the true directional power delta
                raw_power_delta = stack_power_goal_kw - self.previous_stack_power
                
                # FIX 1: Use a small numerical threshold (1 Watt) to eliminate floating-point 1e-14 noise
                if abs(raw_power_delta) > 0.001:
                    # FIX 2: Use absolute value to calculate real durations for both ramp ups AND ramp downs
                    ramp_time_seconds = abs(raw_power_delta) / max_ramp_rate_kw_s
                    
                    # Ensure the calculated ramp duration never exceeds the 10-minute step limit
                    ramp_time_seconds = min(ramp_time_seconds, 600.0)
                    
                    # High-Fidelity Physics Adjustment:
                    # Calculate true integrated average power across the 10-minute window
                    p_avg_during_ramp = (self.previous_stack_power + stack_power_goal_kw) / 2.0
                    p_steady_state = stack_power_goal_kw
                    time_steady = 600.0 - ramp_time_seconds
                    
                    P_stack_average_kw = ((p_avg_during_ramp * ramp_time_seconds) + (p_steady_state * time_steady)) / 600.0
                else:
                    # True steady state operation
                    ramp_time_seconds = 0.0
                    P_stack_average_kw = stack_power_goal_kw
            
            # elif state=="OPERATIONAL":
            dt_seconds = 600.0
            q_net_stack = q_gen_stack - q_loss_stack- q_cooling_stack
            dT = (q_net_stack / PEM.C_TOTAL_J_K) * dt_seconds
            self.current_T += dT
            Power_ratio= P_stack_actual/13.5
            if self.current_T > PEM.T_max:
                print(f"🚨🚨 THERMAL CEILING BREACHED! Core temperature reached {self.current_T:.2f} K (Limit: {PEM.T_max} K).")
                
            self.previous_stack_power = P_stack_actual
               
            # Mass Balance
            # =========================================================================
            # ── PHASE 4: MASS BALANCE & GAS PURITY DIAGNOSTICS ──
            # =========================================================================
            # Theoretical raw production rates based on Faraday's Law
            n_dot_h2_theo = (I_stack * PEM.N_cells) / (MM.z * MM.F)
            n_dot_o2_theo = (I_stack * PEM.N_cells) / (2 * MM.z * MM.F)
            
            # Membrane cross-permeation physics parameters
            Dmem_H2 = MM.D_H2(self.current_T)
            Hmem_H2 = MM.H_H2(self.current_T)
            C_eq = (PEM.p_cathode * 1e5) / Hmem_H2          # Boundary concentration (mol/m^3)
            J_mt = MM.k_mt * PEM.A_m2 * PEM.N_cells         # Transport conductance per unit driving force
            
            # Cross-permeation loss calculation across the solid polymer electrolyte core
            C_H2_mem = (PEM.p_cathode * 1e5) / Hmem_H2
            n_perm_h2_stack = (PEM.A_m2 * PEM.N_cells * Dmem_H2 / PEM.L_pem_m) * C_H2_mem  # Cross-over loss rate (mol/s)
            
            # --- THE STATE PARTITIONING FILTER ---
            if state == "COLD START" or state == "OFF" or state == "STANDBY":
                # Physically, zero net hydrogen is safely collected or exported
                n_dot_h2_gen = 0.0
                n_dot_o2_gen = 0.0
                h2_in_o2_percent = 0.0
                eta_F_physical = 0.0
                h2_gen = 0.0
                
                # AS REQUESTED: Hard-lock Specific Energy Consumption cleanly to 0
                SEC = 0.0
            else:
                # OPERATIONAL MODE: Evaluate exact Faraday efficiency degradations
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
                    
                # Integrated total mass generated over this 10-minute (600 seconds) discrete step (grams)
                h2_gen = n_dot_h2_gen * 600.0 * MM.M_H2*self.N_stacks_total
                
                # Calculate Specific Energy Consumption dynamically (kWh/kg or MJ/kg depending on base metrics)
                # Safeguard against zero division to maintain model stability during edge load cases
                if n_dot_h2_gen > 0:
                    # SEC calculation based on your conversion scaling factor
                    SEC = P_stack_actual / (n_dot_h2_gen * 6.0)
                else:
                    SEC = 0.0
            
            # Record results for this 10-minute interval
            results_list.append({
                "Time": current_time,
                "Status": status,
                "Wind_Power": wind_power,
                "Stack_Power": stack_power,
                "Temperature_K": self.current_T,
                "State": state,
                "J_solved": j_solved,
                "I_stack": I_stack,
                "V_stack": V_stack,
                "V_cell": V_cell,
                "P_stack_actual": P_stack_actual,
                "V_thermoneutral": V_thermoneutral,
                "Cooling_System_Active": cooling_system_active,
                "q_cooling_stack": q_cooling_stack,
                "ramp_time_seconds": ramp_time_seconds,
                "Hydrogen_Generation": h2_gen,
                "SEC": SEC,
                "h2_in_o2_percent": h2_in_o2_percent,
                "Operational_Minutes": operational_minutes_step,
                "Cold_Start_Minutes": cold_start_minutes_step,
                "Power_ratio": Power_ratio
                
                
            })
            
            # Advance clock
            current_time += step_size
        # Convert all results into a final DataFrame
        return pd.DataFrame(results_list)

if __name__ == "__main__":
    sim = ElectrolyzerSimulation(filename="Anholt_hub_analysis.csv", plant_capacity_mw=135.0)
    results_df = sim.run_simulation()
    
    print(results_df.head())
    results_df.to_csv("test_results.csv", index=False)
    # Extract and display the true total cumulative operational hours
    total_op_hours = results_df['Operational_Minutes'].sum() / 60.0
    print(f"True Cumulative Operational Time: {total_op_hours:.2f} Hours")
    is_cold_start = results_df['State'] == 'COLD START'
    was_not_cold_start = results_df['State'].shift(1) != 'COLD START'
    cold_start_events = results_df[is_cold_start & was_not_cold_start].shape[0]

    # 3. Apply the Event Tax (100 microvolts per cold start)
    beta_cold = 100 / 1e6  # 100 microvolts converted to Volts
    Total_cold_start_degradation = cold_start_events * beta_cold
    Total_steady_state_degradation= total_op_hours*2.3/1e6*results_df['Power_ratio'].mean()
    degradation_limit= 0.1*results_df['V_cell'].mean()
    SOH=1- (Total_steady_state_degradation/degradation_limit)
    print(f"SOH: {SOH:.2f}")
    x= 1- SOH
    Total_Lifetime= 1/x
    print(f"Total steady-state Lifetime: {Total_Lifetime:.2f} Years")
    Total_acummulated_degradation= Total_cold_start_degradation+Total_steady_state_degradation
    SOH_acummulated= 1- (Total_acummulated_degradation/degradation_limit)
    y= 1- SOH_acummulated
    Total_Lifetime_acummulated= 1/y
    print(f"Total cold start events: {cold_start_events}")
    print(f"Total accumulated Lifetime: {Total_Lifetime_acummulated:.2f} Years")
    print("Simulation finished successfully!")
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Time'], results_df['Stack_Power'])
    plt.axhline(y=sim.plant_capacity_kw, color='r', linestyle='--', label='Plant Capacity')
    plt.axhline(y=0.1*sim.plant_capacity_kw, color='g', linestyle='--', label='Minimum Stack Power')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stack Power (kW)')
    plt.title('Stack Power vs Time', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Time'], results_df['Temperature_K'])
    plt.axhline(y=PEM.T_max, color='r', linestyle='--', label='Thermal Ceiling')
    plt.axhline(y=PEM.T_min, color='g', linestyle='--', label='Thermal Floor')
    plt.axhline(y=298, color='b', linestyle='--', label='Initial Temperature')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    