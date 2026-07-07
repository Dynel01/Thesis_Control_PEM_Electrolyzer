import pyomo.environ as pyo
import pandas as pd
import numpy as np
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

df_slice = df.iloc[0:144*7].reset_index(drop=True)
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
mode_params = df_soh1.set_index('mode_id')[['Power', 'J_solved', 'V_stack','V_cells', 'I_stack', 'current_T','V_tn', 'h2_gen_rate', 'Degradation_cost']].to_dict('index')
power_map = {i: mode_params[i]['Power'] for i in model.I}
h2_map = {i: mode_params[i]['h2_gen_rate'] for i in model.I}
J_map = {i: mode_params[i]['J_solved'] for i in model.I}
Vs_map = {i: mode_params[i]['V_stack'] for i in model.I}
Vc_map = {i: mode_params[i]['V_cells'] for i in model.I}
i_map = {i: mode_params[i]['I_stack'] for i in model.I}
T_map = {i: mode_params[i]['current_T'] for i in model.I}
vtn_map = {i: mode_params[i]['V_tn'] for i in model.I}
# --- ON-state degradation, per mode -------------------------------------
# Same formula you derived: current-density-driven voltage rise, scaled by
# an Arrhenius-style thermal acceleration term, converted to a per-10-min
# step value (the *(1/6) matches dt_seconds/3600). Computed directly from
# J_map/T_map (already in results_table.csv) -- no need to regenerate the
# lookup table.
B_constant = 0.04
T_nominal_K = 60.0 + 273.15
deg_map = {}
for i in MODES:
    K_thermal = np.exp(B_constant * (T_map[i] - T_nominal_K))
    deg_map[i] = ((0.2499 * J_map[i] + 2.3545) / 1e6) * K_thermal * (1 / 6)

# CAPEX (from Lookup_Table_for_OPT.py: capex_per_kw * stack_capacity_kw * N_stacks_total)
CAPEX_total = 494 * 13.5 * 10000

# Standby degradation rate: Lu et al. 2023, "constant low current" category
# (Table 2, PEM_Degradation_Quantification paper) -- NOT the ON formula,
# since standby is a distinct (OCV, zero-current) physical state.
mu_standby_rate = 1.5e-6  # V/h
mu_standby_step = mu_standby_rate * (dt_seconds / 3600)  # per 10-min step

# --- Heat recovery / district heating revenue, per mode ------------------
T_target_dh = 70.0 + 273.15
dT_pinch = 2.0
eta_hp = 0.45
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
P_COLD = 0.10 * plant_capacity_kw

P_aux = 0.05*plant_capacity_kw  # kW, drawn whenever the plant is not fully OFF

# 135 MW is the true interconnection/nameplate ceiling. The stack itself
# can use up to (135,000 - P_aux); the rest is reserved for auxiliary load
# so the plant never asks the grid connection for more than 135 MW total.
stack_capacity_kw_max = plant_capacity_kw - P_aux
# =========================================================
# 4. CONSTRAINTS
# =========================================================
M = 150000.0
M_temp= 400
T_limit = PEM.T_min # 40°C in Kelvin
T_max = PEM.T_max # 80°C in Kelvin
def exclusivity_rule(m, t):
    return (
        m.y_off[t]
        + m.y_on[t]+m.y_standby[t]+m.y_cold_start[t]
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

def cold_start_power_upper(m, t):
    return m.P_stack[t] <= P_COLD + M * (1 - m.y_cold_start[t])
model.cold_start_power_upper = pyo.Constraint(model.T, rule=cold_start_power_upper)

def cold_start_power_lower(m, t):
    return m.P_stack[t] >= P_COLD - M * (1 - m.y_cold_start[t])
model.cold_start_power_lower = pyo.Constraint(model.T, rule=cold_start_power_lower)

def cold_start_temp_rule(m, t):
    if t == 0:
        return pyo.Constraint.Skip
    T_prev = 298.15 if t == 1 else m.T_stack[t - 1]
    return (
        T_prev <= T_limit + M_temp * (1 - m.y_cold_start[t])
    )
    # T_prev = 298.15 if t == 1 else m.T_stack[t - 1]
    # return (
    #     T_prev <= T_limit + M_temp * (1 - m.y_cold_start[t])
    # )
model.cold_start_temp= pyo.Constraint(model.T, rule=cold_start_temp_rule)

# Standby: zero voltage -> zero current through the stack (no electrochemistry,
# no H2, no stack heat generation -- matches the same Power=0 rows OFF uses).
# The real standby draw (pumps, controls) is P_aux, handled separately in
# grid_def/total_power_cap below, not through the stack at all.
def standby_power_zero(m, t):
    return m.P_stack[t] <= M * (1 - m.y_standby[t])
model.standby_power_zero = pyo.Constraint(model.T, rule=standby_power_zero)

def standby_temp_rule(m, t):
    return (
        m.T_stack[t] >= T_limit - M_temp * (1 - m.y_standby[t])
    )
model.standby_temp= pyo.Constraint(model.T, rule=standby_temp_rule)

def on_power_rule(m, t):
    return (
        m.P_stack[t] >= (0.11 * plant_capacity_kw) * m.y_on[t]
    )
model.on_power= pyo.Constraint(model.T, rule=on_power_rule)

def on_temp_rule(m, t):
    T_prev = 298.15 if t == 0 else m.T_stack[t-1]
    return (
        T_prev >= (T_limit+1) - M_temp * (1 - m.y_on[t])
    )
model.on_temp= pyo.Constraint(model.T, rule=on_temp_rule)

def sequence_rule(m, t):
    if t == 0:
        return pyo.Constraint.Skip
    
    # If the system was OFF at t-1, it is FORBIDDEN to go straight to ON or STANDBY
    # It MUST be OFF or COLD START at t
    return m.y_on[t] <= 1 - m.y_off[t-1] 
    
model.sequence_constraint = pyo.Constraint(model.T, rule=sequence_rule)
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
    
# Linearized replacement for q_cooling[t] <= q_limit_dynamic[t] * y_cool[t].
# That was a bilinear (variable * binary) term -> quadratic constraint.
# Standard big-M split gives the same behavior with a clean MILP:
#   - when y_cool=1: q_cooling <= q_limit_dynamic (the real physical cap)
#   - when y_cool=0: q_cooling <= 0 (forced off), q_limit slack is irrelevant
M_cool = 3e5  # generously above any plausible q_cooling value; tighten if you want a faster solve

def cooling_limit_cap_rule(m, t):
    return m.q_cooling[t] <= m.q_limit_dynamic[t] + M_cool * (1 - m.y_cool[t])
model.cooling_limit_cap = pyo.Constraint(model.T, rule=cooling_limit_cap_rule)

def cooling_limit_lock_rule(m, t):
    return m.q_cooling[t] <= M_cool * m.y_cool[t]
model.cooling_limit_lock = pyo.Constraint(model.T, rule=cooling_limit_lock_rule)

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
    return sum(m.x[t, i] for i in m.I) == 1
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

model.z_shutdown_begin = pyo.Var(model.T, within=pyo.Binary)

def shutdown_event_rule1(m, t):
    if t == 0:
        return m.z_shutdown_begin[t] == 0
    return m.z_shutdown_begin[t] >= m.y_off[t] - m.y_off[t-1]
model.shutdown_event1 = pyo.Constraint(model.T, rule=shutdown_event_rule1)

def shutdown_event_rule2(m, t):
    return m.z_shutdown_begin[t] <= m.y_off[t]
model.shutdown_event2 = pyo.Constraint(model.T, rule=shutdown_event_rule2)

Q_delivered_map = {}   # kW, total plant heat delivered to district heating
W_pump_map = {}        # kW, electricity drawn to run the heat pump

for i in MODES:
    q_gen_i = N_Cells * i_map[i] * (Vc_map[i] - vtn_map[i])  # W, per stack
    T_source_i = T_map[i] - 7.5

    if T_source_i >= T_target_dh - dT_pinch:
        Q_delivered_map[i] = (q_gen_i * 10000) / 1000.0   # kW, direct HX, no pump
        W_pump_map[i] = 0.0
    else:
        COP_i = eta_hp * (T_target_dh / (T_target_dh - T_source_i))
        Q_source_kw = (q_gen_i * 10000) / 1000.0
        W_pump_map[i] = Q_source_kw / (COP_i - 1)
        Q_delivered_map[i] = Q_source_kw + W_pump_map[i]

def heat_delivered_raw_rule(m, t):
    return sum(m.x[t, i] * Q_delivered_map[i] for i in m.I)
model.Q_delivered_raw = pyo.Expression(model.T, rule=heat_delivered_raw_rule)

def pump_power_raw_rule(m, t):
    return sum(m.x[t, i] * W_pump_map[i] for i in m.I)
model.W_pump_raw = pyo.Expression(model.T, rule=pump_power_raw_rule)

M_heat = 1e5
model.Q_delivered = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.W_pump = pyo.Var(model.T, within=pyo.NonNegativeReals)

def q_delivered_cap_rule(m, t):
    return m.Q_delivered[t] <= m.Q_delivered_raw[t] + M_heat * (1 - m.y_cool[t])
model.q_delivered_cap = pyo.Constraint(model.T, rule=q_delivered_cap_rule)

def q_delivered_floor_rule(m, t):
    return m.Q_delivered[t] >= m.Q_delivered_raw[t] - M_heat * (1 - m.y_cool[t])
model.q_delivered_floor = pyo.Constraint(model.T, rule=q_delivered_floor_rule)

def q_delivered_lock_rule(m, t):
    return m.Q_delivered[t] <= M_heat * m.y_cool[t]
model.q_delivered_lock = pyo.Constraint(model.T, rule=q_delivered_lock_rule)

def w_pump_cap_rule(m, t):
    return m.W_pump[t] <= m.W_pump_raw[t] + M_heat * (1 - m.y_cool[t])
model.w_pump_cap = pyo.Constraint(model.T, rule=w_pump_cap_rule)

def w_pump_floor_rule(m, t):
    return m.W_pump[t] >= m.W_pump_raw[t] - M_heat * (1 - m.y_cool[t])
model.w_pump_floor = pyo.Constraint(model.T, rule=w_pump_floor_rule)

def w_pump_lock_rule(m, t):
    return m.W_pump[t] <= M_heat * m.y_cool[t]
model.w_pump_lock = pyo.Constraint(model.T, rule=w_pump_lock_rule)
# ON-state degradation: deg_map[i] is meaningful (nonzero current -> real
# voltage-rise formula) for ON-power-level rows, but the SAME table also
# gets used by COLD_START (different, lower fixed power) and OFF/STANDBY
# (Power=0 rows). We only want this cost applied while genuinely ON, so
# deg_on_cost[t] is forced to equal the raw per-mode value when y_on=1,
# and exactly 0 otherwise -- linear big-M gating, no bilinear x*y_on term.
deg_cost_map = {i: CAPEX_total * (deg_map[i] / 0.18) for i in MODES}

def deg_on_raw_rule(m, t):
    return sum(m.x[t, i] * deg_cost_map[i] for i in m.I)
model.deg_on_raw = pyo.Expression(model.T, rule=deg_on_raw_rule)

model.deg_on_cost = pyo.Var(model.T, within=pyo.NonNegativeReals)
M_deg = 500.0  # comfortably above any plausible per-step €-cost now

def deg_on_cap_rule(m, t):
    return m.deg_on_cost[t] <= m.deg_on_raw[t] + M_deg * (1 - m.y_on[t])
model.deg_on_cap = pyo.Constraint(model.T, rule=deg_on_cap_rule)

def deg_on_floor_rule(m, t):
    return m.deg_on_cost[t] >= m.deg_on_raw[t] - M_deg * (1 - m.y_on[t])
model.deg_on_floor = pyo.Constraint(model.T, rule=deg_on_floor_rule)

def deg_on_lock_rule(m, t):
    return m.deg_on_cost[t] <= M_deg * m.y_on[t]
model.deg_on_lock = pyo.Constraint(model.T, rule=deg_on_lock_rule)
# Total degradation cost per step: ON (current/temperature-driven) +
# STANDBY (OCV-hold, Lu et al. rate) + shutdown (fixed 100uV per event).
# No separate cold-start penalty (matches the paper's ON/OFF-only scope --
# cold start's real cost is the lost H2 revenue during warm-up, already
# captured by the objective without needing an extra term here).
def degradation_cost_rule(m, t):
    on_cost = m.deg_on_cost[t]
    standby_cost = CAPEX_total * (mu_standby_step / 0.18) * m.y_standby[t]
    shutdown_cost = CAPEX_total * (100e-6 / 0.18) * m.z_shutdown_begin[t]
    return on_cost + standby_cost + shutdown_cost
model.degradation_cost = pyo.Expression(model.T, rule=degradation_cost_rule)

# REMOVED temp_link_rule: T_target must stay a free decision variable (the
# optimizer's chosen cooling setpoint), not tied to whichever table row got
# picked for the electrochemistry. Tying them together was the root bug:
# it let T_target track an arbitrary/unrelated row's temperature instead of
# a genuine optimized setpoint, and let electrochemistry (V, I, H2) be read
# from a row whose temperature had no relationship to the real, simulated
# T_stack[t].
#
# In its place: force the *mode* (x[t,i]) to be physically consistent with
# the stack temperature. IMPORTANT: compare against T_stack[t-1] (the known
# temperature entering the step), not T_stack[t] (the temperature produced
# BY that same step's heat generation). Comparing against T_stack[t] creates
# a fixed-point requirement -- "the row I pick must predict a temperature
# change small enough to land back within 0.5C of itself" -- which only has
# a solution at very low power (e.g. cold start's ~6C/step) and is globally
# infeasible at higher power, where a single step can swing 25C->80C. Using
# T_stack[t-1] is standard explicit-Euler: evaluate physics at the known
# starting temperature, let the state evolve to wherever that heat takes it.
mode_temp_tol = 2.501  # ~half the 5°C table spacing (np.linspace(25,80,12))

def T_in_expr(m, t):
    return 298.15 if t == 0 else m.T_stack[t - 1]

def mode_temp_upper_rule(m, t, i):
    return T_map[i] - T_in_expr(m, t) <=  mode_temp_tol + M_temp * (1 - m.x[t, i])
model.mode_temp_upper = pyo.Constraint(model.T, model.I, rule=mode_temp_upper_rule)

def mode_temp_lower_rule(m, t, i):
    return T_in_expr(m, t) - T_map[i] <= mode_temp_tol+ M_temp * (1 - m.x[t, i])
model.mode_temp_lower = pyo.Constraint(model.T, model.I, rule=mode_temp_lower_rule)

def terminal_temp_rule(m):
    last_t = m.T.last()
    prev_t = last_t - 1
    # Don't let the free, consequence-free final step swing more than a
    # modest amount from the step before it -- removes the "dump all
    # remaining heat for free" degeneracy, without artificially forcing
    # a specific end value.
    return m.T_stack[last_t] >= m.T_stack[prev_t] - 5.0
model.terminal_temp = pyo.Constraint(rule=terminal_temp_rule)
# model.deg_link = pyo.Constraint(model.T, rule=lambda m, t: m.cost_degradation_var[t] == sum(m.x[t, i] * deg_map[i] for i in m.I))
# model.q_link = pyo.Constraint(model.T, rule=lambda m, t: m.Q_heat[t] == sum(m.x[t, i] * q_map[i] for i in m.I))
# model.w_pump_link = pyo.Constraint(model.T, rule=lambda m, t: m.W_pump[t] == sum(m.x[t, i] * w_pump_map[i] for i in m.I))

# Detects the FIRST step of each shutdown event (y_off flips 0->1). The
# fixed 100 microvolt penalty (paper's mu_sd, CS4 choice) is charged once per
# genuine shutdown, matching their definition -- NOT on restart/cold-start,
# and not on t=0 (that's just the initial condition, not a real event).
# Min load and Wind limits -- P_aux is drawn whenever the plant isn't fully OFF
# (COLD_START, STANDBY, ON all need pumps/controls running)
def power_limit_rule(m, t):
    return m.P_stack[t] + P_aux * (1 - m.y_off[t]) <= wind_available[t]
model.power_limit = pyo.Constraint(model.T, rule=power_limit_rule)

def grid_def_rule(m, t):
    return m.P_grid[t] == wind_available[t] - m.P_stack[t] - P_aux * (1 - m.y_off[t])
model.grid_def = pyo.Constraint(model.T, rule=grid_def_rule)

def total_power_cap_rule(m, t):
    return m.P_stack[t] + P_aux * (1 - m.y_off[t]) <= plant_capacity_kw
model.total_power_cap = pyo.Constraint(model.T, rule=total_power_cap_rule)
# Force the system to be ON at t=10
print(wind_available[10])
# Force the system into Cold Start at t=1
# TEMPORARY DEBUG PROBE
# model.debug_force_early_on = pyo.Constraint(
#     expr=sum(model.y_on[t] for t in range(1, 100)) >= 50
# )
# =========================================================
# 6. OBJECTIVE
# =========================================================
def objective_rule(model):
    pi_h2 = 5.0

    revenue_h2 = sum(pi_h2 * model.m_h2[t] for t in model.T)
    grid_rev = sum(
        (prices_dk1[t] / 1e3) * (1/6) * model.P_grid[t]
        for t in model.T
    )
    degradation = sum(model.degradation_cost[t] for t in model.T)
    pi_heat = 30.0
    heat_rev = sum(
        model.Q_delivered[t] * (pi_heat/1e3) * (1/6)
        - model.W_pump[t] * (prices_dk1[t]/1e3) * (1/6)
        for t in model.T
    )

    return revenue_h2 + grid_rev - degradation+ heat_rev

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
solver = pyo.SolverFactory('gurobi')
solver.options['MIPGap'] = 0.005   # stop within 0.5% of optimal
solver.options['MIPFocus'] = 1     # prioritize good solutions over proving optimality
# Keep this: if the model is ever infeasible again, Gurobi will auto-compute
# and dump a real IIS here (unlike the old, dead `model.compute_iis = True`).
solver.options['ResultFile'] = 'debug_infeasible.ilp'
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
print("deg_map sample (mode 50):", deg_map[50])
print("deg_on_raw at t=5:", pyo.value(model.deg_on_raw[5]))
print("deg_on_cost at t=5:", pyo.value(model.deg_on_cost[5]))
print("y_on at t=5:", pyo.value(model.y_on[5]))
results_debug=[]
for t in model.T:
    results_debug.append({
    "time": t, 
    "y_on": pyo.value(model.y_on[t]),
    "y_off": pyo.value(model.y_off[t]),
    "y_standby": pyo.value(model.y_standby[t]),
    "y_cold_start": pyo.value(model.y_cold_start[t]),
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
    "Q limit": pyo.value(model.q_limit_dynamic[t]),
    "P_grid": pyo.value(model.P_grid[t]),
    "z_shutdown_begin": pyo.value(model.z_shutdown_begin[t]),
    "deg_on_cost": pyo.value(model.deg_on_cost[t]),
    "degradation_cost": pyo.value(model.degradation_cost[t]),
})
debug_df = pd.DataFrame(results_debug)
debug_df.to_csv("debug3.csv", index=False)
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