# import pandas as pd
# import matplotlib.pyplot as plt

# pi_h2 = 5.0
# pi_h2_grey = 2.0
# pi_heat = 30.0

# cases = {
#     "Pure Revenue": "Simulation_Results/Scenario_1_Volatility/mpc_results_pure_revenue_scenario_1.csv",
#     "Weighted + Floor (Green/Grey) + Shortfall($5/kg)": "Simulation_Results/Scenario_1_Volatility/mpc_results_guaranteed_floor_scenario_1.csv",
#     "Weighted + Floor (Green/Grey) + Shortfall($2/kg)": "Simulation_Results/Scenario_1_Volatility/mpc_results_guaranteed_floor_shortfall_less_scenario_1.csv",
#     "Floor (Green/Grey) + Shortfall($2/kg)": "Simulation_Results/Scenario_1_Volatility/mpc_results_guaranteed_floor_shortfall_less_scenario_1_lower_weight.csv",
#     "Full Horizon MPC": "Simulation_Results/Scenario_1_Volatility/Full_Horizon_MPC_Scenario_1.csv",
#     "Normalized": "Simulation_Results/Scenario_1_Volatility/normalized_results_0.4_scenario_1.csv",
#     "Normalized(More weightage to H2)": "Simulation_Results/Scenario_1_Volatility/normalized_results_0.5_scenario_1.csv"# adjust filename if different
# }

# results = {}

# for label, filename in cases.items():
#     df = pd.read_csv(filename)

#     total_h2 = df['H2'].sum()
#     total_h2_green = df['H2_green'].sum() if 'H2_green' in df else total_h2
#     total_h2_grey = df['H2_grey'].sum() if 'H2_grey' in df else 0.0

#     h2_revenue = pi_h2 * total_h2_green + pi_h2_grey * total_h2_grey
#     grid_revenue = ((df['price'] / 1e3) * (1/6) *
#                      (df['Grid Export'] - df['Grid Import'])).sum()
#     heat_revenue = (df['Q_delivered'] * (pi_heat / 1e3) * (1/6)).sum() - \
#                    (df['W_pump'] * (df['price'] / 1e3) * (1/6)).sum()
#     degradation_cost = df['degradation_cost'].sum()

#     net_total = h2_revenue + grid_revenue + heat_revenue - degradation_cost

#     results[label] = {
#         'H2_total': total_h2/1e3,
#         'Net_Revenue': net_total/1e6
#     }
#     print(f"{label}: H2={total_h2:,.0f} kg, Net Revenue=EUR {net_total:,.0f}")

# # --- Plot ---
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']
# markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

# for (label, vals), color, marker in zip(results.items(), colors, markers):
#     ax.scatter(vals['H2_total'], vals['Net_Revenue'], s=150, color=color,
#                marker=marker, label=label, edgecolors='black',alpha=0.7, zorder=3)
#     # ax.annotate(label, (vals['H2_total'], vals['Net_Revenue']),
#                 # textcoords="offset points", xytext=(10, 10), fontsize=9)

# ax.set_xlabel('Total H2 Delivered (tonnes)')
# ax.set_ylabel('Net Revenue (Million EUR)')
# ax.set_title('Revenue vs. Hydrogen Delivery Trade-off Across Objective Formulations')
# ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.axvline(60, color='k', linestyle='--', lw=1, alpha=0.5)
# # plt.savefig("case_comparison_scatter.png", dpi=150)
# plt.show()

# totals = {}
# for label, f in cases.items():
#     df = pd.read_csv(f)
#     totals[label] = df['degradation_cost'].sum()/1e6

# fig, ax = plt.subplots(figsize=(9,5))
# ax.bar(totals.keys(), totals.values(), color='red', edgecolor='black')
# ax.set_ylabel('Total Degradation Cost (Million EUR)')
# ax.set_title('Degradation Cost Across Objective Formulations')
# plt.xticks(rotation=30, ha='right')
# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter

pi_h2 = 5.0
pi_h2_grey = 2.0
pi_heat = 30.0

stack_replacement_cost_per_kw = 494.0  # EUR/kW
plant_capacity_kw = 135000.0
CAPEX_stack_total = stack_replacement_cost_per_kw * plant_capacity_kw  # EUR, full plant
window_days = 3.0  # each scenario is a 3-day window

scenarios = [
    ("Scenario_1_Volatility", "Volatility", "1"),
    ("Scenario_2_Scarcity", "Scarcity", "2"),
    ("Scenario_3_Cannibalization", "Cannibalization", "3"),
    ("Scenario_4_Generic", "Generic", "4"),
]

FORMULATION_LABELS = [
    "Pure Revenue",
    "Weighted + Floor (Green/Grey) + Shortfall($5/kg)",
    "Weighted + Floor (Green/Grey) + Shortfall($2/kg)",
    "Floor (Green/Grey) + Shortfall($2/kg)",
    "Full Horizon MPC",
    "Normalized",
    "Normalized(More weightage to H2)",
]

def build_cases(folder, num):
    base = f"Simulation_Results/{folder}"
    return {
        "Pure Revenue": f"{base}/mpc_results_pure_revenue_scenario_{num}.csv",
        "Weighted + Floor (Green/Grey) + Shortfall($5/kg)": f"{base}/mpc_results_guaranteed_floor_scenario_{num}.csv",
        "Weighted + Floor (Green/Grey) + Shortfall($2/kg)": f"{base}/mpc_results_guaranteed_floor_shortfall_less_scenario_{num}.csv",
        "Floor (Green/Grey) + Shortfall($2/kg)": f"{base}/mpc_results_guaranteed_floor_shortfall_less_scenario_{num}_lower_weight.csv",
        "Full Horizon MPC": f"{base}/Full_Horizon_MPC_Scenario_{num}.csv",
        "Normalized": f"{base}/normalized_results_0.4_scenario_{num}.csv",
        "Normalized(More weightage to H2)": f"{base}/normalized_results_0.5_scenario_{num}.csv",
    }

def compute_results(cases):
    results = {}
    for label, filename in cases.items():
        df = pd.read_csv(filename)

        total_h2 = df['H2'].sum()
        total_h2_green = df['H2_green'].sum() if 'H2_green' in df else total_h2
        total_h2_grey = df['H2_grey'].sum() if 'H2_grey' in df else 0.0

        h2_revenue = pi_h2 * total_h2_green + pi_h2_grey * total_h2_grey
        grid_revenue = ((df['price'] / 1e3) * (1 / 6) *
                         (df['Grid Export'] - df['Grid Import'])).sum()
        heat_revenue = (df['Q_delivered'] * (pi_heat / 1e3) * (1 / 6)).sum() - \
                       (df['W_pump'] * (df['price'] / 1e3) * (1 / 6)).sum()
        degradation_cost = df['degradation_cost'].sum()

        net_total = h2_revenue + grid_revenue + heat_revenue - degradation_cost

        if degradation_cost > 0:
            stack_life_years = (CAPEX_stack_total / degradation_cost) * (window_days / 365.0)
        else:
            stack_life_years = float('inf')

        results[label] = {
            'H2_total': total_h2 / 1e3,          # tonnes
            'Net_Revenue': net_total / 1e6,       # million EUR
            'Degradation_Cost': degradation_cost / 1e6,  # million EUR
            'Stack_Life_Years': stack_life_years
        }
    return results

colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

all_results = {}
for folder, title, num in scenarios:
    print(f"\n--- {title} ---")
    cases = build_cases(folder, num)
    all_results[title] = compute_results(cases)
    for label, vals in all_results[title].items():
        life_str = "inf" if vals['Stack_Life_Years'] == float('inf') else f"{vals['Stack_Life_Years']:.1f} yr"
        print(f"  {label}: H2={vals['H2_total']:.2f} t, "
              f"Net Revenue=EUR {vals['Net_Revenue']:.3f}M, "
              f"Stack Life={life_str}")

# --- Figure 1: 2x2 grid of H2 vs Revenue scatter, improved separation for duplicates ---
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 11))
axes1 = axes1.flatten()

for ax, (folder, title, num) in zip(axes1, scenarios):
    results = all_results[title]
    items = list(results.items())
    xs_raw = [v['H2_total'] for _, v in items]
    ys_raw = [v['Net_Revenue'] for _, v in items]

    x_min, x_max = min(xs_raw), max(xs_raw)
    y_min, y_max = min(ys_raw), max(ys_raw)
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    n = len(items)
    cluster_id = list(range(n))
    for a in range(n):
        for b in range(a + 1, n):
            dx = abs(xs_raw[a] - xs_raw[b]) / x_range
            dy = abs(ys_raw[a] - ys_raw[b]) / y_range
            if dx < 0.08 and dy < 0.08:
                root_a, root_b = cluster_id[a], cluster_id[b]
                for k in range(n):
                    if cluster_id[k] == root_b:
                        cluster_id[k] = root_a

    clusters = {}
    for idx, c in enumerate(cluster_id):
        clusters.setdefault(c, []).append(idx)

    x_spread = max(x_range, 15)
    y_spread = max(y_range, 0.005)

    plotted_xy = {}
    for c_id, members in clusters.items():
        k = len(members)
        if k == 1:
            i = members[0]
            plotted_xy[i] = (xs_raw[i], ys_raw[i])
        else:
            cx = sum(xs_raw[i] for i in members) / k
            cy = sum(ys_raw[i] for i in members) / k
            radius_x = 0.08 * x_spread * (0.6 + 0.4 * k)
            radius_y = 0.08 * y_spread * (0.6 + 0.4 * k)
            for pos, i in enumerate(members):
                angle = 2 * math.pi * pos / k
                jx = cx + radius_x * math.cos(angle)
                jy = cy + radius_y * math.sin(angle)
                plotted_xy[i] = (jx, jy)

    for i, (label, vals) in enumerate(items):
        x, y = plotted_xy[i]
        ax.scatter(x, y, s=160, color=colors[i], marker=markers[i],
                   edgecolors='black', alpha=0.9, zorder=3)
        ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                    xytext=(0, 10), fontsize=9, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
                    zorder=4)

    ax.axvline(60, color='black', alpha=0.3, linestyle='--', lw=2)
    ax.set_xlabel('Total H2 Delivered (tonnes)')
    ax.set_ylabel('Net Revenue (Million EUR)')
    ax.set_title(title)

    ax.set_xlim(x_min - 0.15 * x_range - 2, x_max + 0.15 * x_range + 2)
    ax.set_ylim(y_min - 0.15 * y_range - 0.002, y_max + 0.15 * y_range + 0.002)

    ax.grid(True, alpha=0.3)

legend_handles = [plt.Line2D([0], [0], marker=markers[i], color='w',
                            markerfacecolor=colors[i], markeredgecolor='black',
                            markersize=10, label=f"{i + 1}. {FORMULATION_LABELS[i]}")
                   for i in range(len(FORMULATION_LABELS))]
fig1.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=8,
            bbox_to_anchor=(0.5, -0.09), frameon=True)
fig1.suptitle('Revenue vs. Hydrogen Delivery Trade-off Across Formulations and Scenarios', fontsize=14)
plt.tight_layout(rect=[0, 0.10, 1, 0.96])
plt.savefig('scenario_comparison_h2_revenue.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 2: 2x2 grid of STACK LIFE (years), numbered x-axis matching Figure 1 ---
STACK_LIFE_CAP = 25  # visual cap when a formulation shows zero degradation (infinite implied life)

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
axes2 = axes2.flatten()

for ax, (folder, title, num) in zip(axes2, scenarios):
    results = all_results[title]
    life_values = []
    label_text = []
    for l in FORMULATION_LABELS:
        v = results[l]['Stack_Life_Years']
        if v == float('inf'):
            life_values.append(STACK_LIFE_CAP)
            label_text.append('∞')
        elif v > STACK_LIFE_CAP:
            life_values.append(STACK_LIFE_CAP)
            label_text.append(f'{v:.1f}')
        else:
            life_values.append(v)
            label_text.append(None)

    bars = ax.bar(range(1, len(FORMULATION_LABELS) + 1), life_values, color=colors, edgecolor='black')
    for bar, txt in zip(bars, label_text):
        if txt is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    txt, ha='center', fontsize=9, fontweight='bold')

    ax.axhline(20, color='black', linestyle='--', lw=1, alpha=0.4)  # 20-year project horizon
    ax.set_xticks(range(1, len(FORMULATION_LABELS) + 1))
    ax.set_ylabel('Implied Stack Life (years)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

fig2.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=8,
            bbox_to_anchor=(0.5, -0.09), frameon=True)
fig2.suptitle('Implied Stack Life Across Formulations and Scenarios\n(dashed line = 20-year project horizon; \u221e = zero measured degradation in this window)', fontsize=13)
plt.tight_layout(rect=[0, 0.10, 1, 0.94])
plt.savefig('scenario_comparison_stack_life.png', dpi=150, bbox_inches='tight')
plt.show()