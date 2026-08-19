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
# 1. DATA (loaded once)
# =========================================================
df = pd.read_csv("Anholt_hub_analysis.csv")
results_df = pd.read_csv("results_table.csv")

step_size = timedelta(minutes=10)
dt_seconds = step_size.seconds

C_Total = PEM.C_TOTAL_J_K
N_Cells = PEM.N_cells
LSA = TM.LSA
h_total = TM.h_total_ins
T_amb = TM.T_ambient

plant_capacity_kw = 135000.0
M = 150000.0
M_temp = 400
M_cool = 3e5
M_deg = 500.0
T_limit = PEM.T_min
T_max = PEM.T_max
P_COLD = 0.10 * plant_capacity_kw
P_aux = 0.05 * plant_capacity_kw
stack_capacity_kw_max = plant_capacity_kw - P_aux
mode_temp_tol = 2.501  # ~half the 5C table spacing
cumulative_h2_delivered = 0.0
# =========================================================
# 2. MODE TABLE (SOH = 1 ONLY) -- built once, reused by every window
# =========================================================
df_soh1 = results_df[results_df['current_SOH'].round(1) == 1.0].copy()
df_soh1['mode_id'] = range(len(df_soh1))
print(f"DEBUG: Rows found for SOH 1.0: {len(df_soh1)}")
MODES = df_soh1['mode_id'].tolist()

mode_params = df_soh1.set_index('mode_id')[
    ['Power', 'J_solved', 'V_stack', 'V_cells', 'I_stack', 'current_T', 'V_tn', 'h2_gen_rate']
].to_dict('index')

power_map = {i: mode_params[i]['Power'] for i in MODES}
h2_map = {i: mode_params[i]['h2_gen_rate'] for i in MODES}
J_map = {i: mode_params[i]['J_solved'] for i in MODES}
Vc_map = {i: mode_params[i]['V_cells'] for i in MODES}
i_map = {i: mode_params[i]['I_stack'] for i in MODES}
T_map = {i: mode_params[i]['current_T'] for i in MODES}
vtn_map = {i: mode_params[i]['V_tn'] for i in MODES}

# ON-state degradation, per mode
B_constant = 0.04
T_nominal_K = 60.0 + 273.15
deg_map = {}
for i in MODES:
    K_thermal = np.exp(B_constant * (T_map[i] - T_nominal_K))
    deg_map[i] = ((0.2499 * J_map[i] + 2.3545) / 1e6) * K_thermal * (1 / 6)

CAPEX_total = 494 * 13.5 * 10000
deg_cost_map = {i: CAPEX_total * (deg_map[i] / 0.18) for i in MODES}

mu_standby_rate = 1.5e-6  # V/h, Lu et al. 2023 "constant low current"
mu_standby_step = mu_standby_rate * (dt_seconds / 3600)

# Heat recovery / district heating revenue, per mode
T_target_dh = 70.0 + 273.15
dT_pinch = 2.0
eta_hp = 0.45

Q_delivered_map = {}
W_pump_map = {}
for i in MODES:
    q_gen_i = N_Cells * i_map[i] * (Vc_map[i] - vtn_map[i])
    T_source_i = T_map[i] - 7.5
    if T_source_i >= T_target_dh - dT_pinch:
        Q_delivered_map[i] = (q_gen_i * 10000) / 1000.0
        W_pump_map[i] = 0.0
    else:
        COP_i = eta_hp * (T_target_dh / (T_target_dh - T_source_i))
        Q_source_kw = (q_gen_i * 10000) / 1000.0
        W_pump_map[i] = Q_source_kw / (COP_i - 1)
        Q_delivered_map[i] = Q_source_kw + W_pump_map[i]

pi_h2 = 5.0
pi_h2_grey= 2.0
pi_heat = 30.0


# =========================================================
# 3. BUILD + SOLVE ONE MPC WINDOW
# =========================================================
def build_and_solve_window(T_stack_init, P_stack_init, y_off_prev, y_cold_start_prev,
                            price_win, wind_win, remaining_needed=0.0,
                            mip_gap=0.01, verbose=True):
    """
    Builds and solves one receding-horizon window.
    T_stack_init, P_stack_init, y_off_prev: real state carried over from the
        previously committed step (fixed numbers, not variables).
    price_win, wind_win: dicts keyed 0..window_len-1 for THIS window only.
    epsilon: max fraction of the window's own H2-maximum the second stage is
        allowed to sacrifice in exchange for heat revenue / lower degradation.
    Returns the solved model (after stage 2) and the window length.
    """
    window_len = len(price_win)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, window_len - 1)
    model.I = pyo.Set(initialize=MODES)

    model.x = pyo.Var(model.T, model.I, within=pyo.Binary)
    model.y_on = pyo.Var(model.T, within=pyo.Binary)
    model.y_off = pyo.Var(model.T, within=pyo.Binary)
    model.y_standby = pyo.Var(model.T, within=pyo.Binary)
    model.y_cold_start = pyo.Var(model.T, within=pyo.Binary)
    model.P_stack = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0.0, plant_capacity_kw))
    model.q_cooling = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0, None))
    model.m_h2 = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.T_stack = pyo.Var(model.T, bounds=(298.15, 353.15), within=pyo.NonNegativeReals)
    model.T_target = pyo.Var(model.T, bounds=(298.15, 353.15), within=pyo.NonNegativeReals)
    model.y_cool = pyo.Var(model.T, within=pyo.Binary)
    model.P_grid_import = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.P_grid_export = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.y_grid_direction = pyo.Var(model.T, within=pyo.Binary)
    model.h2_green = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.h2_grey = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.h2_shortfall = pyo.Var( model.T, within=pyo.NonNegativeReals)
    model.z_shutdown_begin = pyo.Var(model.T, within=pyo.Binary)
    # Initial conditions -- carried over from the previous committed step
    # (or the true absolute start, for the very first window).
    # t=0 is the fixed, known "current state" -- exactly matching the
    # original script's convention. We commit t=1 (the first genuinely
    # free decision), not t=0, so this anchor stays a pure boundary
    # condition, never something the optimizer needs to decide.
    model.init_power = pyo.Constraint(expr=model.P_stack[0] == P_stack_init)
    model.init_temp = pyo.Constraint(expr=model.T_stack[0] == T_stack_init)

    def exclusivity_rule(m, t):
        return m.y_off[t] + m.y_on[t] + m.y_standby[t] + m.y_cold_start[t] == 1
    model.exclusivity = pyo.Constraint(model.T, rule=exclusivity_rule)

    def off_power_rule(m, t):
        return m.P_stack[t] <= M * (1 - m.y_off[t])
    model.off_power = pyo.Constraint(model.T, rule=off_power_rule)

    def cold_start_power_upper(m, t):
        return m.P_stack[t] <= P_COLD + M * (1 - m.y_cold_start[t])
    model.cold_start_power_upper = pyo.Constraint(model.T, rule=cold_start_power_upper)

    def cold_start_power_lower(m, t):
        return m.P_stack[t] >= P_COLD - M * (1 - m.y_cold_start[t])
    model.cold_start_power_lower = pyo.Constraint(model.T, rule=cold_start_power_lower)

    def T_prev_of(m, t):
        return T_stack_init if t == 0 else m.T_stack[t - 1]

    def cold_start_temp_rule(m, t):
        return T_prev_of(m, t) <= T_limit + M_temp * (1 - m.y_cold_start[t])
    model.cold_start_temp = pyo.Constraint(model.T, rule=cold_start_temp_rule)

    def standby_power_zero(m, t):
        return m.P_stack[t] <= M * (1 - m.y_standby[t])
    model.standby_power_zero = pyo.Constraint(model.T, rule=standby_power_zero)

    def standby_temp_rule(m, t):
        return m.T_stack[t] >= T_limit - M_temp * (1 - m.y_standby[t])
    model.standby_temp = pyo.Constraint(model.T, rule=standby_temp_rule)

    def on_power_rule(m, t):
        return m.P_stack[t] >= (0.10 * plant_capacity_kw) * m.y_on[t]
    model.on_power = pyo.Constraint(model.T, rule=on_power_rule)

    def on_temp_rule(m, t):
        return T_prev_of(m, t) >= (T_limit) - M_temp * (1 - m.y_on[t])
    model.on_temp = pyo.Constraint(model.T, rule=on_temp_rule)

    def sequence_rule(m, t):
        prev_off = y_off_prev if t == 0 else m.y_off[t - 1]
        return m.y_on[t] <= 1 - prev_off
    model.sequence_constraint = pyo.Constraint(model.T, rule=sequence_rule)
    
    def cold_start_requires_prior_off(m, t):
            prev_off = y_off_prev if t == 0 else m.y_off[t - 1]
            prev_cold_start = y_cold_start_prev if t == 0 else m.y_cold_start[t - 1]
            return m.y_cold_start[t] <= prev_off+prev_cold_start
    model.cold_start_requires_prior_off = pyo.Constraint(model.T, rule=cold_start_requires_prior_off)

    def power_ramp_rule(m, t):
        P_prev = P_stack_init if t == 0 else m.P_stack[t - 1]
        max_ramp_allowed = (0.10 * plant_capacity_kw) * dt_seconds
        return m.P_stack[t] - P_prev <= max_ramp_allowed
    model.power_ramp = pyo.Constraint(model.T, rule=power_ramp_rule)

    def q_net_rule(m, t):
        T_prev = T_prev_of(m, t)
        q_gen = sum(m.x[t, i] * (N_Cells * i_map[i] * (Vc_map[i] - vtn_map[i])) for i in m.I)
        q_loss = LSA * h_total * (T_prev - T_amb)
        return q_gen - q_loss
    model.q_net = pyo.Expression(model.T, rule=q_net_rule)

    def delta_T_eff_rule(m, t):
        return m.T_stack[t] - m.T_target[t]
    model.delta_T_eff = pyo.Expression(model.T, rule=delta_T_eff_rule)

    def q_limit_expression(m, t):
        return TM.U * TM.Plate_area * (m.delta_T_eff[t] + 10) * N_Cells
    model.q_limit_dynamic = pyo.Expression(model.T, rule=q_limit_expression)

    def thermal_balance_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.T_stack[t] == m.T_stack[t - 1] + (
            (m.q_net[t] - m.q_cooling[t]) * dt_seconds / C_Total
        )
    model.thermal_balance = pyo.Constraint(model.T, rule=thermal_balance_rule)

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
        return m.T_stack[t] - m.T_target[t] >= -M_temp * (1 - m.y_cool[t])
    model.force_off_if_cold = pyo.Constraint(model.T, rule=force_off_if_cold)

    def cooling_state_coupling(m, t):
        return m.y_cool[t] <= m.y_on[t]
    model.cooling_state_coupling = pyo.Constraint(model.T, rule=cooling_state_coupling)

    def one_mode_per_step(m, t):
        return sum(m.x[t, i] for i in m.I) == 1
    model.one_mode = pyo.Constraint(model.T, rule=one_mode_per_step)

    def power_link_rule(m, t):
        return m.P_stack[t] == sum(m.x[t, i] * power_map[i] for i in m.I)
    model.power_link = pyo.Constraint(model.T, rule=power_link_rule)

    def h2_link_rule(m, t):
        return m.m_h2[t] == dt_seconds * sum(m.x[t, i] * h2_map[i] for i in m.I)
    model.h2_link = pyo.Constraint(model.T, rule=h2_link_rule)

    def h2_split_rule(m, t):
        return m.h2_green[t] + m.h2_grey[t] == m.m_h2[t]
    model.h2_split = pyo.Constraint(model.T, rule=h2_split_rule)
    
    M_h2= 500
    def h2_grey_gate_rule(m, t):
    # grey only allowed when importing that step (y_grid_direction=0)
        return m.h2_grey[t] <= M_h2 * (1 - m.y_grid_direction[t])
    model.h2_grey_gate = pyo.Constraint(model.T, rule=h2_grey_gate_rule)
    
    def h2_green_gate_rule(m, t):
        return m.h2_green[t] <= M_h2 * m.y_grid_direction[t]
    model.h2_green_gate = pyo.Constraint(model.T, rule=h2_green_gate_rule)
    
    def h2_min_rule(m):
        return sum(m.h2_green[t] for t in m.T) + sum(m.h2_shortfall[t] for t in m.T) >= remaining_needed
    model.h2_min_constraint = pyo.Constraint(rule=h2_min_rule)
    
    def shutdown_event_rule1(m, t):
        prev_off = y_off_prev if t == 0 else m.y_off[t - 1]
        return m.z_shutdown_begin[t] >= m.y_off[t] - prev_off
    model.shutdown_event1 = pyo.Constraint(model.T, rule=shutdown_event_rule1)

    def shutdown_event_rule2(m, t):
        return m.z_shutdown_begin[t] <= m.y_off[t]
    model.shutdown_event2 = pyo.Constraint(model.T, rule=shutdown_event_rule2)
    
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

    def deg_on_raw_rule(m, t):
        return sum(m.x[t, i] * deg_cost_map[i] for i in m.I)
    model.deg_on_raw = pyo.Expression(model.T, rule=deg_on_raw_rule)

    model.deg_on_cost = pyo.Var(model.T, within=pyo.NonNegativeReals)

    def deg_on_cap_rule(m, t):
        return m.deg_on_cost[t] <= m.deg_on_raw[t] + M_deg * (1 - m.y_on[t])
    model.deg_on_cap = pyo.Constraint(model.T, rule=deg_on_cap_rule)

    def deg_on_floor_rule(m, t):
        return m.deg_on_cost[t] >= m.deg_on_raw[t] - M_deg * (1 - m.y_on[t])
    model.deg_on_floor = pyo.Constraint(model.T, rule=deg_on_floor_rule)

    def deg_on_lock_rule(m, t):
        return m.deg_on_cost[t] <= M_deg * m.y_on[t]
    model.deg_on_lock = pyo.Constraint(model.T, rule=deg_on_lock_rule)

    def degradation_cost_rule(m, t):
        on_cost = m.deg_on_cost[t]
        standby_cost = CAPEX_total * (mu_standby_step / 0.18) * m.y_standby[t]
        shutdown_cost = CAPEX_total * (100e-6 / 0.18) * m.z_shutdown_begin[t]
        return on_cost + standby_cost + shutdown_cost
    model.degradation_cost = pyo.Expression(model.T, rule=degradation_cost_rule)

    def mode_temp_upper_rule(m, t, i):
        return T_map[i] - T_prev_of(m, t) <= mode_temp_tol + M_temp * (1 - m.x[t, i])
    model.mode_temp_upper = pyo.Constraint(model.T, model.I, rule=mode_temp_upper_rule)

    def mode_temp_lower_rule(m, t, i):
        return T_prev_of(m, t) - T_map[i] <= mode_temp_tol + M_temp * (1 - m.x[t, i])
    model.mode_temp_lower = pyo.Constraint(model.T, model.I, rule=mode_temp_lower_rule)

    def terminal_temp_rule(m):
        last_t = m.T.last()
        prev_t = last_t - 1
        return m.T_stack[last_t] >= m.T_stack[prev_t] - 5.0
    model.terminal_temp = pyo.Constraint(rule=terminal_temp_rule)
    
    M_grid = plant_capacity_kw + 5e5
    def grid_import_cap_rule(m, t):
        return m.P_grid_import[t] <= plant_capacity_kw
    model.grid_import_cap = pyo.Constraint(model.T, rule=grid_import_cap_rule)

    def import_lock_rule(m, t):
        return m.P_grid_import[t] <= M_grid * (1 - m.y_grid_direction[t])
    model.import_lock = pyo.Constraint(model.T, rule=import_lock_rule)

    def export_lock_rule(m, t):
        return m.P_grid_export[t] <= M_grid * m.y_grid_direction[t]
    model.export_lock = pyo.Constraint(model.T, rule=export_lock_rule)
    def power_limit_rule(m, t):
        return m.P_stack[t] + P_aux * (1 - m.y_off[t]) <= wind_win[t]+m.P_grid_import[t]
    model.power_limit = pyo.Constraint(model.T, rule=power_limit_rule)

    def grid_def_rule(m, t):
        return m.P_grid_import[t]+wind_win[t] == (m.P_stack[t] + P_aux * (1 - m.y_off[t])+ m.P_grid_export[t])
    model.grid_def = pyo.Constraint(model.T, rule=grid_def_rule)

    def total_power_cap_rule(m, t):
        return m.P_stack[t] + P_aux * (1 - m.y_off[t]) <= plant_capacity_kw
    model.total_power_cap = pyo.Constraint(model.T, rule=total_power_cap_rule)
    
    # model.h2_pacing_constraint = pyo.Constraint(
    # expr=sum(model.m_h2[t] for t in range(1, CONTROL_HORIZON+1)) >= remaining_needed)

    # -----------------------------------------------------
    # STAGE 1: maximize H2 alone (the primary objective).
    # -----------------------------------------------------
    # discount = {t: 0.97 ** t for t in range(window_len)}

    # model.Objective_H2 = pyo.Objective(
    #     expr=sum(discount[t] * model.m_h2[t] for t in model.T), sense=pyo.maximize
    # )

    # solver = pyo.SolverFactory('gurobi')
    # solver.options['MIPFocus'] = 1

    # # Stage 1 is a much simpler problem (pure H2 sum only).
    # solver.options['MIPGap'] = 0.0005
    # solver.solve(model, tee=verbose)

    # # Record STAGE 1's actual per-step schedule. This
    # # is what Stage 2 gets compared against, timestep by timestep.
    # h2_stage1 = {t: pyo.value(model.m_h2[t]) for t in model.T}
    # H2_max = sum(h2_stage1.values())

    # -----------------------------------------------------
    # STAGE 2: at EVERY timestep, stay within epsilon of what Stage 1
    # already decided for that specific step -- not just the window total.
    # -----------------------------------------------------
    # def h2_floor_rule(m, t):
    #     return m.m_h2[t] >= (1 - epsilon) * h2_stage1[t]
    # model.h2_floor = pyo.Constraint(model.T, rule=h2_floor_rule)
    # model.Objective_H2.deactivate()

    def objective_rule(m):
        revenue_h2 = 1.5 * (sum(pi_h2 * m.h2_green[t] for t in m.T))+sum(pi_h2_grey * m.h2_grey[t] for t in m.T)
        grid_rev = sum((price_win[t] / 1e3) * (1 / 6) * (m.P_grid_export[t]-m.P_grid_import[t]) for t in m.T)
        degradation = sum( m.degradation_cost[t] for t in m.T)
        heat_rev = sum(
             (
                m.Q_delivered[t] * (pi_heat / 1e3) * (1 / 6)
                - m.W_pump[t] * (price_win[t] / 1e3) * (1 / 6)
            )
            for t in m.T
        )
        pure_revenue= revenue_h2 + grid_rev - degradation + heat_rev
        h2_shortfall= sum(pi_h2_grey*m.h2_shortfall[t] for t in m.T)
        return pure_revenue - h2_shortfall
    model.Objective_secondary = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['DualReductions'] = 0
    solver.options['MIPFocus'] = 1
    solver.options['MIPGap'] = mip_gap  # back to the looser, speed-focused gap for the harder problem
    results=solver.solve(model, tee=verbose)
    if str(results.solver.termination_condition) == 'infeasible':
        print("INFEASIBLE — computing IIS to find the exact conflicting constraints...")
        solver.options['ResultFile'] = 'infeasible_model.ilp'
        solver.solve(model, tee=True, options={'ResultFile': 'infeasible_model.ilp'})
        print("Check infeasible_model.ilp for the exact constraints in conflict.")
    print(f"  Status: {results.solver.termination_condition}, "
      f"Gap at cutoff: {getattr(results.solver, 'gap', 'n/a')}")
    return model


# =========================================================
# 4. MPC ROLLING LOOP
# =========================================================
WINDOW_LEN = 144          # 24 hours at 10-min resolution -- prediction horizon
BASE_IDX = 0          # 2014-01-11 12:00 -- start of the demonstration period
TOTAL_REAL_STEPS = 51337    # 3 days 
CONTROL_HORIZON = 143       # commit 6 steps (1 hour) before re-solving, cuts
H_MIN_PER_DAY = 20000
H_min = H_MIN_PER_DAY * (TOTAL_REAL_STEPS / 144)         
# EPSILON = 0.02            # allow at most 2% H2 sacrifice for heat/degradation

T_stack_current = 298.15
P_stack_current = 0.0
y_off_current = 1.0
y_cold_start_current = 0.0
# Real time 0 (== BASE_IDX in the underlying data) is the true, known
# simulation start -- trivially OFF, no solve needed for it.
committed = [{
    'time': 0, 'P_stack': 0.0, 'T_stack': 298.15, 'H2': 0.0,
    'Q_delivered': 0.0, 'W_pump': 0.0, 'degradation_cost': 0.0,
    'y_on': 0.0, 'y_off': 1.0, 'y_standby': 0.0, 'y_cold_start': 0.0,
    'wind': float(df.loc[BASE_IDX, 'wtc_ActPower_mean']),
    'Grid Import': 0.0, 'Grid Export': 0.0,
    'price': float(df.loc[BASE_IDX, 'SpotPrice_DK1']) * 3,
    'window_H2_max': None, 'active_mode': None,
}]

k = 0
while k < TOTAL_REAL_STEPS:
    price_win = {t: float(df.loc[BASE_IDX + k + t, 'SpotPrice_DK1']) * 3 for t in range(WINDOW_LEN)}
    wind_win = {t: float(df.loc[BASE_IDX + k + t, 'wtc_ActPower_mean']) for t in range(WINDOW_LEN)}
    required_by_now = H_min * (k + CONTROL_HORIZON) / TOTAL_REAL_STEPS
    remaining_needed = max(0, required_by_now - cumulative_h2_delivered)*0.8

    model = build_and_solve_window(
        T_stack_current, P_stack_current, y_off_current, y_cold_start_current,
        price_win, wind_win, remaining_needed=remaining_needed, verbose=False)

    # Commit CONTROL_HORIZON steps (1..CONTROL_HORIZON)
    for c in range(1, CONTROL_HORIZON + 1):
        active_mode = None
        for i in model.I:
            if pyo.value(model.x[c, i]) > 0.5:
                active_mode = i
                break

        committed.append({
            'time': k + c,
            'P_stack': pyo.value(model.P_stack[c]),
            'P_stack_plus_aux': pyo.value(model.P_stack[c]) + P_aux * (1 - pyo.value(model.y_off[c])),
            'T_stack': pyo.value(model.T_stack[c]),
            'H2': pyo.value(model.m_h2[c]),
            'H2_green': pyo.value(model.h2_green[c]),
            'H2_grey': pyo.value(model.h2_grey[c]),
            'Q_delivered': pyo.value(model.Q_delivered[c]),
            'W_pump': pyo.value(model.W_pump[c]),
            'degradation_cost': pyo.value(model.degradation_cost[c]),
            'y_on': pyo.value(model.y_on[c]),
            'y_off': pyo.value(model.y_off[c]),
            'y_standby': pyo.value(model.y_standby[c]),
            'y_cold_start': pyo.value(model.y_cold_start[c]),
            'wind': wind_win[c],
            'Grid Import': pyo.value(model.P_grid_import[c]),
            'Grid Export': pyo.value(model.P_grid_export[c]),
            'price': price_win[c],
            'active_mode': active_mode,
        })

    # Carry forward from the LAST committed step (index CONTROL_HORIZON)
    T_stack_current = pyo.value(model.T_stack[CONTROL_HORIZON])
    P_stack_current = pyo.value(model.P_stack[CONTROL_HORIZON])
    y_off_current = pyo.value(model.y_off[CONTROL_HORIZON])
    y_cold_start_current = pyo.value(model.y_cold_start[CONTROL_HORIZON])

    print(f"[real t={k+1}-{k+CONTROL_HORIZON}] committed P={committed[-1]['P_stack']:.0f} kW, "
          f"T={committed[-1]['T_stack']:.2f} K, H2={committed[-1]['H2']:.2f} kg")
    committed_h2_this_window = sum(pyo.value(model.m_h2[t]) for t in range(1, CONTROL_HORIZON+1))
    cumulative_h2_delivered += committed_h2_this_window
    k += CONTROL_HORIZON
    

mpc_results = pd.DataFrame(committed)
mpc_results.to_csv("mpc_results_annual_gf15.csv", index=False)
num_solves = -(-TOTAL_REAL_STEPS // CONTROL_HORIZON)
print(f"\nDone. {TOTAL_REAL_STEPS} real steps committed via {num_solves} solves, results in mpc_results.csv")
print(f"Total H2 (committed steps): {mpc_results['H2'].sum():.2f} kg")
print(f"Total degradation cost (committed steps): EUR {mpc_results['degradation_cost'].sum():,.2f}")
print(f"Gap at cutoff: {mpc_results.solver.gap if hasattr(mpc_results.solver,'gap') else 'n/a'}")