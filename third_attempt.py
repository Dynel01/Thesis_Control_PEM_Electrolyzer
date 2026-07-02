import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import Electrochemical_Model as ECM
import Thermal_Model as TM
import Mass_balance_model as MM
import PEMWE_Parameters as PEM
# =========================================================
# 1. DATA
# =========================================================
model = pyo.ConcreteModel()

df = pd.read_csv("Anholt_hub_analysis.csv")
results_df = pd.read_csv("results_table.csv")

df_slice = df.iloc[0:24].reset_index(drop=True)
# base_price = df_slice['SpotPrice_DK1'].mean()
start_time = datetime(2014, 1, 1, 0, 0)
model.T = pyo.RangeSet(0, len(df_slice) - 1)
step_size = timedelta(minutes=10)
dt_seconds = step_size.seconds

prices_dk1 = {t: float(df_slice.loc[t, 'SpotPrice_DK1']*3) for t in model.T}
# prices_dk1 = {
#     t: 4.0 * (
#         base_price +
#         3.0 * (float(df_slice.loc[t, 'SpotPrice_DK1']) - base_price)
#     )
#     for t in model.T
# }
wind_available = {t: float(df_slice.loc[t, 'wtc_ActPower_mean']) for t in model.T}
C_Total= PEM.C_TOTAL_J_K
N_Cells= PEM.N_cells
LSA = TM.LSA
h_total = TM.h_total_ins
T_amb = TM.T_ambient
# =========================================================
# 2. MODE TABLE (SOH = 1 ONLY)
# =========================================================
df_soh1 = results_df[results_df['current_SOH'].round(1) == 1.0].copy()
df_soh1['mode_id'] = range(len(df_soh1))
print(f"DEBUG: Rows found for SOH 1.0: {len(df_soh1)}")
MODES = df_soh1['mode_id'].tolist()
model.I = pyo.Set(initialize=MODES)


# Lookup dictionaries
mode_params = df_soh1.set_index('mode_id')[['Power', 'J_solved', 'V_stack','V_cells', 'I_stack', 'current_T','V_tn', 'h2_gen_rate']].to_dict('index')
power_map = {i: mode_params[i]['Power'] for i in model.I}
h2_map = {i: mode_params[i]['h2_gen_rate'] for i in model.I}
J_map = {i: mode_params[i]['J_solved'] for i in model.I}
Vs_map = {i: mode_params[i]['V_stack'] for i in model.I}
Vc_map = {i: mode_params[i]['V_cells'] for i in model.I}
i_map = {i: mode_params[i]['I_stack'] for i in model.I}
T_map = {i: mode_params[i]['current_T'] for i in model.I}
vtn_map = {i: mode_params[i]['V_tn'] for i in model.I}
# =========================================================
# 3. VARIABLES
# =========================================================
plant_capacity_kw = 135000.0
model.x = pyo.Var(model.T, model.I, within=pyo.Binary)
model.y_on = pyo.Var(model.T, within=pyo.Binary)
model.y_off = pyo.Var(model.T, within=pyo.Binary)
model.y_standby = pyo.Var(model.T, within=pyo.Binary)
model.y_cold_start = pyo.Var(model.T, within=pyo.Binary)
model.P_stack = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0.0*plant_capacity_kw, plant_capacity_kw))
model.P_grid = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.q_cooling = pyo.Var(model.T, within=pyo.NonNegativeReals,bounds=(0, None))
model.m_h2 = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.T_stack = pyo.Var(model.T, bounds=(298.15, 353.15), within=pyo.NonNegativeReals)
model.T_target = pyo.Var(model.T, bounds=(298.15, 353.15), within=pyo.NonNegativeReals)
model.y_cool = pyo.Var(model.T, within=pyo.Binary)
# Initial conditions
model.init_power = pyo.Constraint(expr=model.P_stack[0] == 0.0 * plant_capacity_kw)
model.init_temp  = pyo.Constraint(expr=model.T_stack[0] == 298.15)

# Also anchor the state to OFF at t=0 so the solver doesn't start in a random state
P_STANDBY = 0.05 * plant_capacity_kw
P_COLD = 0.10 * plant_capacity_kw
# =========================================================
# 4. CONSTRAINTS
# =========================================================
M = 200000.0
M_temp= 400
T_limit = PEM.T_min # 40°C in Kelvin
T_max = PEM.T_max # 80°C in Kelvin
def exclusivity_rule(m, t):
    return (
        m.y_off[t]
        + m.y_on[t]
        == 1
    )
model.exclusivity = pyo.Constraint(model.T, rule=exclusivity_rule)
    
def off_power_rule(m, t):
    return (
        m.P_stack[t] <= M * (1 - m.y_off[t])
    )
model.off_power= pyo.Constraint(model.T, rule=off_power_rule)

# def off_temp_rule(m, t):
#     return (
#         m.T_stack[t] <= T_limit + M_temp * (1 - m.y_off[t])
#     )
# model.off_temp= pyo.Constraint(model.T, rule=off_temp_rule)

# def cold_start_power_rule(m, t):
#     return (
#         m.P_stack[t] == P_COLD * m.y_cold_start[t]
#     )
# model.cold_start_power= pyo.Constraint(model.T, rule=cold_start_power_rule)

# def cold_start_temp_rule(m, t):
#     if t == 0:
#         return pyo.Constraint.Skip
#     return (
#         m.T_stack[t] <= T_limit + M_temp * (1 - m.y_cold_start[t])
#     )
# model.cold_start_temp= pyo.Constraint(model.T, rule=cold_start_temp_rule)

# def standby_power_rule(m, t):
#     return (
#         m.P_stack[t] == P_STANDBY * m.y_standby[t]
#     )
# model.standby_power= pyo.Constraint(model.T, rule=standby_power_rule)

# def standby_temp_rule(m, t):
#     return (
#         m.T_stack[t] >= T_limit - M_temp * (1 - m.y_standby[t])
#     )
# model.standby_temp= pyo.Constraint(model.T, rule=standby_temp_rule)

def on_power_rule(m, t):
    return (
        m.P_stack[t] >= (0.1 * plant_capacity_kw) * m.y_on[t]
    )
model.on_power= pyo.Constraint(model.T, rule=on_power_rule)

def on_temp_rule(m, t):
    return (
        m.T_stack[t] >= T_limit - M_temp * (1 - m.y_on[t])
    )
model.on_temp= pyo.Constraint(model.T, rule=on_temp_rule)

# def sequence_rule(m, t):
#     if t == 0:
#         return pyo.Constraint.Skip
    
#     # If the system was OFF at t-1, it is FORBIDDEN to go straight to ON or STANDBY
#     # It MUST be OFF or COLD START at t
#     return m.y_on[t] + m.y_standby[t] <= 1 - m.y_off[t-1] 
    
# model.sequence_constraint = pyo.Constraint(model.T, rule=sequence_rule)
def power_ramp_rule(m, t):
    if t == 0: 
        return pyo.Constraint.Skip
    
    # 10% per second * duration of transition (in seconds)
    # This says: "The power change cannot exceed the ramp rate times the time spent in transition"
    max_ramp_allowed = (0.10 * plant_capacity_kw) * dt_seconds 
    
    return m.P_stack[t] - m.P_stack[t-1] <= max_ramp_allowed
model.power_ramp = pyo.Constraint(model.T, rule=power_ramp_rule)

def q_net_rule(m, t):
    if t == 0:
        return 0.0 
    # This sums up the heat generation for whichever mode (i) the solver picks at time (t)
    q_gen = sum(m.x[t, i] * (N_Cells * i_map[i] * (Vc_map[i] - vtn_map[i])) for i in m.I)
    q_loss = LSA * h_total * (m.T_stack[t-1] - T_amb)
    return (q_gen - q_loss)

model.q_net = pyo.Expression(model.T, rule=q_net_rule)

# Replace your current delta_T_eff definition with this:
def delta_T_eff_rule(m, t):
    return m.T_stack[t] - m.T_target[t]
model.delta_T_eff = pyo.Expression(model.T, rule=delta_T_eff_rule)

def q_limit_expression(m, t):
    # TM.T_water_inlet is your fixed cooling source temperature
    return TM.U * TM.Plate_area * (m.delta_T_eff[t]+10) * N_Cells

model.q_limit_dynamic = pyo.Expression(model.T, rule=q_limit_expression)

# def total_time_rule(m, t):
#     return m.d_cold[t] + m.d_trans[t] + m.d_on[t] == dt_seconds
# model.total_time = pyo.Constraint(model.T, rule=total_time_rule)

def thermal_balance_rule(m, t):
    if t == 0:
        return pyo.Constraint.Skip

    return m.T_stack[t] == m.T_stack[t-1] + (
        (m.q_net[t] - m.q_cooling[t]) * dt_seconds / C_Total
    )
model.thermal_balance = pyo.Constraint(model.T, rule=thermal_balance_rule)
    
def cooling_limit_rule(m, t):
    return m.q_cooling[t] <= m.q_limit_dynamic[t]*m.y_cool[t]

model.cooling_limit = pyo.Constraint(model.T, rule=cooling_limit_rule)

def cooling_active_rule(m, t):
    return m.T_stack[t] - m.T_target[t] <= M_temp * m.y_cool[t]
model.cooling_active = pyo.Constraint(model.T, rule=cooling_active_rule)

def force_off_if_cold(m, t):
    # This prevents the pump from turning on if T < T_target
    # If y_cool is 1, T must be >= T_target
    return m.T_stack[t] - m.T_target[t] >= -M_temp * (1 - m.y_cool[t])
model.force_off_if_cold = pyo.Constraint(model.T, rule=force_off_if_cold)

# Force cooling binary to be strictly tied to stack state
def cooling_state_coupling(m, t):
    # This prevents y_cool from being 1 if y_on is 0
    return m.y_cool[t] <= m.y_on[t]
model.cooling_state_coupling = pyo.Constraint(model.T, rule=cooling_state_coupling)

def one_mode_per_step(m, t):
    # If the system is ON, it must pick one mode from the table
    return sum(m.x[t, i] for i in m.I) == m.y_on[t]+m.y_off[t]
model.one_mode = pyo.Constraint(model.T, rule=one_mode_per_step)

def power_link_rule(m, t):
    return m.P_stack[t] == sum(m.x[t, i] * power_map[i] for i in m.I)
model.power_link = pyo.Constraint(model.T, rule=power_link_rule)

def h2_link_rule(m, t):
    return m.m_h2[t] == dt_seconds * sum(
    m.x[t,i] * h2_map[i]
    for i in m.I
)
model.h2_link = pyo.Constraint(model.T, rule=h2_link_rule)

def temp_link_rule(m, t):
   return m.T_target[t] == sum(m.x[t, i] * T_map[i] for i in m.I)
model.temp_link = pyo.Constraint(model.T, rule=temp_link_rule)
# model.deg_link = pyo.Constraint(model.T, rule=lambda m, t: m.cost_degradation_var[t] == sum(m.x[t, i] * deg_map[i] for i in m.I))
# model.q_link = pyo.Constraint(model.T, rule=lambda m, t: m.Q_heat[t] == sum(m.x[t, i] * q_map[i] for i in m.I))
# model.w_pump_link = pyo.Constraint(model.T, rule=lambda m, t: m.W_pump[t] == sum(m.x[t, i] * w_pump_map[i] for i in m.I))

# Min load and Wind limits
model.power_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_stack[t] <= wind_available[t] * m.y_on[t])
model.grid_def = pyo.Constraint(model.T, rule=lambda m, t: m.P_grid[t] == wind_available[t] - m.P_stack[t])
# Force the system to be ON at t=10
print(wind_available[10])
# Force the system into Cold Start at t=1

# =========================================================
# 6. OBJECTIVE
# =========================================================
def objective_rule(model):
    pi_h2, pi_heat = 5.0, 30.0

    revenue_h2 = sum(pi_h2 * model.m_h2[t] for t in model.T)
    # degradation = sum( model.cost_degradation_var[t] for t in model.T)

    # heat_rev = sum(
    #      (model.Q_heat[t] * pi_heat * (1/6))
    #     - (model.W_pump[t] * prices_dk1[t] * (1/6))
    #     for t in model.T)
    # # )

    grid_rev = sum(
        (prices_dk1[t] / 1e3) * (1/6) * model.P_grid[t]
        for t in model.T
    )
    # stack_rev = sum(
    #     (prices_dk1[t] / 1e3) * (1/6) * model.P_stack[t]
    #     for t in model.T
    # )

    # return revenue_h2 + grid_rev - degradation + heat_rev
    # return revenue_h2 + grid_rev - degradation+ heat_rev
    # return revenue_h2 -stack_rev - degradation
    return revenue_h2 + grid_rev

model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

# =========================================================
# 7. SOLVE
# =========================================================
# model.thermal_balance.deactivate()
# model.cooling_limit.deactivate()
# model.power_ramp.deactivate() # If it exists

# # 2. Relax the State Lock
# if hasattr(model, 'init_state'):
#     model.init_state.deactivate()

# # 3. Simplify the Objective to a pure Revenue Maximizer
# # (Avoids complex heat/pump math that might be zeroed out)
# model.Objective.deactivate()
# model.Objective_simple = pyo.Objective(expr=sum(5000.0 * model.m_h2[t] for t in model.T), sense=pyo.maximize)
# model.thermal_balance.deactivate()
# model.cooling_limit.deactivate()
# model.power_ramp.deactivate()
# model.off_power.deactivate()
# model.on_temp.deactivate()
# 4. Solve


# Then run the IIS to see why it refuses to turn on
model.compute_iis = True
solver = pyo.SolverFactory('gurobi')
results = solver.solve(model, tee=True)

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Optimization Successful!")
    # Output your data here...
else:
    print("STILL INFEASIBLE. CHECK YOUR POWER CONSTRAINTS.")
revenue = pyo.value(model.Objective)
print(f"Profit:€{revenue:,.2f}")
# =========================================================
# 8. OUTPUT
# =========================================================
results_data = []
for t in model.T:
    # Identify which mode was active
    active_mode = None
    for i in model.I:
        if pyo.value(model.x[t, i]) > 0.5:
            active_mode = i
            break
            
    results_data.append({
        'Time': t,
        'Power': pyo.value(model.P_stack[t]),
        'Wind': wind_available[t],
        'H2_Gen': pyo.value(model.m_h2[t]),
        'Temperature': pyo.value(model.T_stack[t]) - 273.15, # Convert K to °C
        'State_ON': pyo.value(model.y_on[t]),
        'Active_Mode': active_mode
    })

df_res = pd.DataFrame(results_data)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Power and Hydrogen Generation
ax1.plot(df_res['Time'], df_res['Power'], label='Stack Power (kW)', color='blue')
ax1.plot(df_res['Time'], df_res['Wind'], label='Wind Available (kW)', color='green')
ax1.axhline(y=plant_capacity_kw, color='green', linestyle='--', label='Max Plant Capacity (kW)')
ax1.axhline(y=0.1*plant_capacity_kw, color='red', linestyle='--', label='Max Wind Available (kW)')
ax1.set_ylabel('Power (kW)')
ax1.legend(loc='upper left')

# Plot 2: Temperature and Operational State
ax2.plot(df_res['Time'], df_res['Temperature'], label='Stack Temp (°C)', color='red')
ax2.fill_between(df_res['Time'], 0, df_res['State_ON'] * 100, color='green', alpha=0.2, label='ON State')
ax2.set_ylabel('Temp (°C)')
ax2.set_xlabel('Time Step (10 min)')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
results_debug=[]
for t in model.T:
    results_debug.append({
    "time": t, 
    "y_on": pyo.value(model.y_on[t]),
    "y_off": pyo.value(model.y_off[t]),
    "y_cool": pyo.value(model.y_cool[t]),
    "P": pyo.value(model.P_stack[t]),
    "wind": wind_available[t],
    "Spot Price": prices_dk1[t],
    "H2": pyo.value(model.m_h2[t]),
    "sum x": sum(pyo.value(model.x[t,i]) for i in model.I),
    "Q cooling": pyo.value(model.q_cooling[t]),
    "T": pyo.value(model.T_stack[t]),
    "Target T": pyo.value(model.T_target[t]),
    "Temp difference": pyo.value(model.T_stack[t]) - pyo.value(model.T_target[t]),
    "Delta T eff": pyo.value(model.delta_T_eff[t]),
    "Q net": pyo.value(model.q_net[t]),
    "Q limit": pyo.value(model.q_limit_dynamic[t])     
})
debug_df = pd.DataFrame(results_debug)
debug_df.to_csv("debug.csv", index=False)
# # Create a list to store hourly results
# results = []
# for t in model.T:
#     # Find which mode (if any) is active
#     active_mode = None
#     for i in model.I:
#         if pyo.value(model.x[t, i]) > 0.5:
#             active_mode = i
            
#     results.append({
#         'Timestamp': t,
#         'Status': 'ON' if pyo.value(model.y_on[t]) > 0.5 else 'OFF',
#         'Temperature': selected_temps[t],
#         'Active_Mode': active_mode,
#         'Stack_Power': pyo.value(model.P_stack[t]),
#         'H2_Gen': pyo.value(model.m_h2[t]),
#         'Spot_Price': pyo.value(prices_dk1[t]),
#         'wind': wind_available[t],
#         # 'Q_Heat': pyo.value(model.Q_heat[t]),
#         'Degradation': pyo.value(model.cost_degradation_var[t])
#     })

# # Convert to DataFrame and export
# df_out = pd.DataFrame(results)
# df_out.to_excel("optimization_debugging.xlsx")
# # 1. Calculate profit for EVERY row in the CSV
# pi_h2, pi_heat, cop = 5.0, 30.0, 3.75
# price_dk1 = 30.0 # Using a sample price

# # Use your CSV data directly
# df_eval = df_soh1.copy()

# # Apply the exact same Profit formula as your Objective Function
# def calculate_row_profit(row):
#     h2_rev = pi_h2 * row['h2_gen']
#     deg_cost = row['degradation_cost']
#     q_val = row['q_delivered']
#     # Heat revenue calculation
#     heat_rev = (q_val * 1e-2 * pi_heat * (1/6)) - \
#                (q_val * 1e-2 / (cop - 1) * (price_dk1) * (1/6))
#     return h2_rev - deg_cost + heat_rev

# df_eval['profit'] = df_eval.apply(calculate_row_profit, axis=1)

# # 2. Group by temperature to see which is BEST
# best_profit_by_temp = df_eval.groupby('current_T')['profit'].mean()
# print("--- ACTUAL PROFITABILITY BY TEMP ---")
# print(best_profit_by_temp)

# for p in [13500, 27000, 54000, 81000, 108000, 135000]:
#     subset = df_soh1[df_soh1['Power'] == p]

#     print("\nPower =", p)

#     subset = subset.copy()

#     subset['profit'] = subset.apply(calculate_row_profit, axis=1)

#     print(
#         subset[['current_T','profit',
#                 'h2_gen',
#                 'degradation_cost',
#                 'q_delivered']]
#         .sort_values('profit', ascending=False)
#     )
#     # Check if Mode 181 is actually the best profit at every power level
# for p in [13500, 27000, 54000, 81000, 108000, 135000]:
#     best_row = df_eval[df_eval['Power'] == p].sort_values('profit', ascending=False).iloc[0]
#     print(f"At Power {p}, the best Temp is {best_row['current_T']} K")
    