import pyomo.environ as pyo
import pandas as pd
import numpy as np
from datetime import timedelta
import Electrochemical_Model as ECM
import Thermal_Model as TM
import Mass_balance_model as MM
import PEMWE_Parameters as PEM

# =========================================================
# 1. DATA + MODE TABLE
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
P_COLD = 0.10 * plant_capacity_kw
P_aux = 0.05 * plant_capacity_kw
mode_temp_tol = 2.501

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

B_constant = 0.04
T_nominal_K = 60.0 + 273.15
deg_map = {}
for i in MODES:
    K_thermal = np.exp(B_constant * (T_map[i] - T_nominal_K))
    deg_map[i] = ((0.2499 * J_map[i] + 2.3545) / 1e6) * K_thermal * (1 / 6)

CAPEX_total = 494 * 13.5 * 10000
deg_cost_map = {i: CAPEX_total * (deg_map[i] / 0.18) for i in MODES}

mu_standby_rate = 1.5e-6
mu_standby_step = mu_standby_rate * (dt_seconds / 3600)

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
pi_h2_grey = 2.0
pi_heat = 30.0

# Sensible degradation reference: nominal-mode steady operation, NOT the
# pathological worst-case cycling scenario. This is the fix for the
# nadir-point degeneracy problem found in the earlier normalization attempt.
nominal_mode_id = min(MODES, key=lambda i: abs(T_map[i] - T_nominal_K))
DEG_REFERENCE_PER_STEP = deg_cost_map[nominal_mode_id]


# =========================================================
# 2. SHARED MODEL BUILDER (constraints only, no objective)
# =========================================================
def build_model(T_stack_init, P_stack_init, y_off_prev, price_win, wind_win, remaining_needed=0.0):
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
    model.h2_shortfall = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.z_shutdown_begin = pyo.Var(model.T, within=pyo.Binary)

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
        prev_cold_start = 0 if t == 0 else m.y_cold_start[t - 1]
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

    M_h2 = 500
    def h2_grey_gate_rule(m, t):
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
        return m.P_stack[t] + P_aux * (1 - m.y_off[t]) <= wind_win[t] + m.P_grid_import[t]
    model.power_limit = pyo.Constraint(model.T, rule=power_limit_rule)

    def grid_def_rule(m, t):
        return m.P_grid_import[t] + wind_win[t] == (m.P_stack[t] + P_aux * (1 - m.y_off[t]) + m.P_grid_export[t])
    model.grid_def = pyo.Constraint(model.T, rule=grid_def_rule)

    def total_power_cap_rule(m, t):
        return m.P_stack[t] + P_aux * (1 - m.y_off[t]) <= plant_capacity_kw
    model.total_power_cap = pyo.Constraint(model.T, rule=total_power_cap_rule)

    return model


# =========================================================
# 3. NORMALIZATION REFERENCE SOLVES + FINAL WEIGHTED SCORE
# =========================================================
def solve_weighted_normalized(T_stack_init, P_stack_init, y_off_prev,
                               price_win, wind_win, remaining_needed=0.0,
                               weights=(0.5, 0.3, 0.1, 0.1),  # (H2, Grid, Heat, Degradation)
                               pi_shortfall=2.0,
                               mip_gap=0.0001, verbose=False):
    window_len = len(price_win)
    w_h2, w_grid, w_heat, w_deg = weights
    discount = {t: 0.99 ** t for t in range(window_len)}

    def solve_single_objective(expr_fn, label):
        m = build_model(T_stack_init, P_stack_init, y_off_prev, price_win, wind_win, remaining_needed)
        m.Obj = pyo.Objective(expr=expr_fn(m), sense=pyo.maximize)
        solver = pyo.SolverFactory('gurobi')
        solver.options['MIPGap'] = mip_gap
        solver.options['MIPFocus'] = 1
        results = solver.solve(m, tee=verbose)
        print(f"  [{label}] status: {results.solver.termination_condition}")
        return m

    # --- Reference 1: max H2 (green, weighted toward priority) alone ---
    # m_h2max = solve_single_objective(
    #     lambda m: sum(discount[t] * (pi_h2 * m.h2_green[t] + pi_h2_grey * m.h2_grey[t]) for t in m.T),
    #     "H2_max ref"
    # )
    # H2_max = sum(pi_h2 * pyo.value(m_h2max.h2_green[t]) + pi_h2_grey * pyo.value(m_h2max.h2_grey[t]) for t in m_h2max.T)
    # H2_max = max(H2_max, 1e-6)
    H2_max = H_min * pi_h2  # fixed, target-based reference -- revenue value of hitting the mandatory floor
    H2_max = max(H2_max, 1e-6)

    # --- Reference 2: max grid revenue alone ---
    def grid_expr(m):
        return sum(discount[t] * (price_win[t]/1e3) * (1/6) * (m.P_grid_export[t]-m.P_grid_import[t]) for t in m.T)
    m_gridmax = solve_single_objective(grid_expr, "Grid_max ref")
    Grid_max = sum((price_win[t]/1e3)*(1/6)*(pyo.value(m_gridmax.P_grid_export[t])-pyo.value(m_gridmax.P_grid_import[t])) for t in m_gridmax.T)
    Grid_max = max(Grid_max, 1e-6)

    # --- Reference 3: max heat revenue alone ---
    def heat_expr(m):
        return sum(discount[t] * (
            m.Q_delivered[t]*(pi_heat/1e3)*(1/6) - m.W_pump[t]*(price_win[t]/1e3)*(1/6)
        ) for t in m.T)
    m_heatmax = solve_single_objective(heat_expr, "Heat_max ref")
    Heat_max = sum(pyo.value(m_heatmax.Q_delivered[t])*(pi_heat/1e3)*(1/6) - pyo.value(m_heatmax.W_pump[t])*(price_win[t]/1e3)*(1/6) for t in m_heatmax.T)
    Heat_max = max(Heat_max, 1e-6)

    # --- Reference 4: degradation -- SENSIBLE reference, not pathological worst-case ---
    Deg_reference = DEG_REFERENCE_PER_STEP * window_len
    Deg_reference = max(Deg_reference, 1e-6)

    print(f"\n  Normalization references: H2_max={H2_max:.2f}, Grid_max={Grid_max:.2f}, "
          f"Heat_max={Heat_max:.2f}, Deg_reference={Deg_reference:.2f} (all EUR)")
    print(f"  NOTE: H2_max and Grid_max are independently-optimal and likely mutually exclusive "
          f"(running the plant vs exporting freely) -- known ideal-point infeasibility limitation.\n")

    # --- Final: weighted, normalized Score ---
    model = build_model(T_stack_init, P_stack_init, y_off_prev, price_win, wind_win, remaining_needed)

    def score_rule(m):
        h2_norm = sum(pi_h2*m.h2_green[t] + pi_h2_grey*m.h2_grey[t] for t in m.T) / H2_max
        grid_norm = sum((price_win[t]/1e3)*(1/6)*(m.P_grid_export[t]-m.P_grid_import[t]) for t in m.T) / Grid_max
        heat_norm = sum(m.Q_delivered[t]*(pi_heat/1e3)*(1/6) - m.W_pump[t]*(price_win[t]/1e3)*(1/6) for t in m.T) / Heat_max
        deg_norm = sum(m.degradation_cost[t] for t in m.T) / Deg_reference
        shortfall_term = sum(pi_shortfall * m.h2_shortfall[t] for t in m.T) / H2_max
        return w_h2*h2_norm + w_grid*grid_norm + w_heat*heat_norm - w_deg*deg_norm - shortfall_term

    model.Score = pyo.Objective(rule=score_rule, sense=pyo.maximize)
    solver = pyo.SolverFactory('gurobi')
    solver.options['MIPGap'] = mip_gap
    solver.options['MIPFocus'] = 1
    results = solver.solve(model, tee=True)
    print(f"  [Final weighted Score] status: {results.solver.termination_condition}")

    total_h2_green = sum(pyo.value(model.h2_green[t]) for t in model.T)
    total_h2_grey = sum(pyo.value(model.h2_grey[t]) for t in model.T)
    total_h2_rev = pi_h2*total_h2_green + pi_h2_grey*total_h2_grey
    total_grid_rev = sum((price_win[t]/1e3)*(1/6)*(pyo.value(model.P_grid_export[t])-pyo.value(model.P_grid_import[t])) for t in model.T)
    total_heat_rev = sum(pyo.value(model.Q_delivered[t])*(pi_heat/1e3)*(1/6) - pyo.value(model.W_pump[t])*(price_win[t]/1e3)*(1/6) for t in model.T)
    total_deg = sum(pyo.value(model.degradation_cost[t]) for t in model.T)

    print(f"\n=== NORMALIZED WEIGHTED OBJECTIVE (weights H2={w_h2}, Grid={w_grid}, Heat={w_heat}, Deg={w_deg}) ===")
    print(f"  Score achieved: {pyo.value(model.Score):.4f}")
    print(f"  H2 green: {total_h2_green:.2f} kg, H2 grey: {total_h2_grey:.2f} kg")
    print(f"  H2 revenue (true, unweighted): EUR {total_h2_rev:,.2f}")
    print(f"  Grid revenue: EUR {total_grid_rev:,.2f}")
    print(f"  Heat revenue: EUR {total_heat_rev:,.2f}")
    print(f"  Degradation: EUR {total_deg:,.2f}")
    print(f"  NET total (true revenue): EUR {total_h2_rev+total_grid_rev+total_heat_rev-total_deg:,.2f}")

    return model, {'H2_max': H2_max, 'Grid_max': Grid_max, 'Heat_max': Heat_max, 'Deg_reference': Deg_reference}


if __name__ == "__main__":
    BASE_IDX = 26208
    WINDOW_LEN = 432
    H_MIN_PER_DAY = 20000
    H_min = H_MIN_PER_DAY * (WINDOW_LEN / 144)

    price_win = {t: float(df.loc[BASE_IDX + t, 'SpotPrice_DK1']) * 3 for t in range(WINDOW_LEN)}
    wind_win = {t: float(df.loc[BASE_IDX + t, 'wtc_ActPower_mean']) for t in range(WINDOW_LEN)}

    model,refs= solve_weighted_normalized(298.15, 0.0, 1.0, price_win, wind_win,
                               remaining_needed=H_min, weights=(0.5, 0.3, 0.1, 0.1),
                               pi_shortfall=2.0, verbose=False)
    
    
    
        # =========================================================
    # CSV EXPORT -- same output format used across the other formulation scripts
    # =========================================================
    rows = []
    for t in model.T:
        active_mode = None
        for i in model.I:
            if pyo.value(model.x[t, i]) > 0.5:
                active_mode = i
                break

        rows.append({
            'time': t,
            'P_stack': pyo.value(model.P_stack[t]),
            'P_stack_plus_aux': pyo.value(model.P_stack[t]) + P_aux * (1 - pyo.value(model.y_off[t])),
            'T_stack': pyo.value(model.T_stack[t]),
            'H2': pyo.value(model.m_h2[t]),
            'H2_green': pyo.value(model.h2_green[t]),
            'H2_grey': pyo.value(model.h2_grey[t]),
            'H2_shortfall': pyo.value(model.h2_shortfall[t]),
            'Q_delivered': pyo.value(model.Q_delivered[t]),
            'W_pump': pyo.value(model.W_pump[t]),
            'degradation_cost': pyo.value(model.degradation_cost[t]),
            'y_on': pyo.value(model.y_on[t]),
            'y_off': pyo.value(model.y_off[t]),
            'y_standby': pyo.value(model.y_standby[t]),
            'y_cold_start': pyo.value(model.y_cold_start[t]),
            'wind': wind_win[t],
            'Grid Import': pyo.value(model.P_grid_import[t]),
            'Grid Export': pyo.value(model.P_grid_export[t]),
            'price': price_win[t],
            'active_mode': active_mode,
        })

    normalized_results = pd.DataFrame(rows)
    normalized_results.to_csv("normalized_results_0.5_scenario_4.csv", index=False)

    total_h2_green = normalized_results['H2_green'].sum()
    total_h2_grey = normalized_results['H2_grey'].sum()
    total_deg = normalized_results['degradation_cost'].sum()

    print(f"\nDone. {WINDOW_LEN} steps solved (normalized, single-shot).")
    print(f"Total H2 green: {total_h2_green:.2f} kg")
    print(f"Total H2 grey: {total_h2_grey:.2f} kg")
    print(f"Total degradation cost: EUR {total_deg:,.2f}")
    print(f"Normalization references used: {refs}")