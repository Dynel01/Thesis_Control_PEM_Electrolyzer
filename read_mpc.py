import pandas as pd
import matplotlib.pyplot as plt

plant_capacity_kw = 135000.0
min_operating_kw = 0.10 * plant_capacity_kw  # matches P_COLD / ON threshold

mpc_results = pd.read_csv("mpc_results_annual_gf10.csv")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax1.step(mpc_results['time'], mpc_results['P_stack_plus_aux'], where='post', label='Stack Power (kW)', color='blue')
ax1.step(mpc_results['time'], mpc_results['wind'], where='post', label='Wind Available (kW)', color='green')
ax1.axhline(plant_capacity_kw, color='green', linestyle='--', label='Max Plant Capacity (kW)')
ax1.axhline(min_operating_kw, color='red', linestyle='--', label='Min Operating Capacity (kW)')
ax1.set_ylabel('Power (kW)')
ax1.legend(loc='upper right')

temp_c = mpc_results['T_stack'] - 273.15
ax2.step(mpc_results['time'], temp_c, where='post', color='red', label='Stack Temp (°C)')
on_mask = mpc_results['y_on'] == 1
cold_start_mask = mpc_results['y_cold_start'] == 1
standby_mask = mpc_results['y_standby'] == 1

ax2.fill_between(
    mpc_results['time'], 0, 100, where=cold_start_mask,
    color='blue', alpha=0.2, label='Cold Start', step='post'
)
ax2.fill_between(
    mpc_results['time'], 0, 100, where=standby_mask,
    color='yellow', alpha=0.2, label='Standby', step='post'
)
ax2.fill_between(
    mpc_results['time'], 0, 100, where=on_mask,
    color='green', alpha=0.2, label='ON State', step='post'
)
ax2.fill_between(
    mpc_results['time'], 0, 100, where=~(on_mask | cold_start_mask | standby_mask),
    color='red', alpha=0.2, label='OFF State', step='post'
)
ax2.set_ylabel('Temp (°C)')
ax2.set_xlabel('Time Step (10 min)')
ax2.set_ylim(0, 100)
ax2.axhline(40, color='black', linestyle='--', linewidth=1)
ax2.axhline(80, color='black', linestyle='--', linewidth=1)

ax2.legend(loc='lower center', fontsize='small', ncols=3)

ax3.plot(mpc_results['time'], mpc_results['price'], label='Price (€/MWh)', color='purple')
ax3.set_ylabel('Price (€/MWh)')
ax3.set_xlabel('Time Step (10 min)')
ax3.legend(loc='upper left')
plt.tight_layout()
# plt.savefig("mpc_results_plot.png", dpi=150)
print("Plot saved to mpc_results_plot.png")
plt.show()

# =========================================================
# REVENUE / COST SUMMARY (matches objective_rule terms)
# =========================================================
pi_h2 = 5.0
pi_h2_grey = 2.0
pi_heat = 30.0

total_h2 = mpc_results['H2'].sum()
total_h2_green = mpc_results['H2_green'].sum() if 'H2_green' in mpc_results else total_h2
total_h2_grey = mpc_results['H2_grey'].sum() if 'H2_grey' in mpc_results else 0.0

h2_revenue =  pi_h2 * total_h2_green + pi_h2_grey * total_h2_grey  # no grey price distinction if none used
grid_revenue = ((mpc_results['price'] / 1e3) * (1/6) *
                 (mpc_results['Grid Export'] - mpc_results['Grid Import'])).sum()
heat_revenue = (mpc_results['Q_delivered'] * (pi_heat / 1e3) * (1/6)).sum() - \
               (mpc_results['W_pump'] * (mpc_results['price'] / 1e3) * (1/6)).sum()
degradation_cost = mpc_results['degradation_cost'].sum()

net_total = h2_revenue + grid_revenue + heat_revenue - degradation_cost

print("\n===== SUMMARY =====")
print(f"Total H2 delivered:      {total_h2:,.2f} kg")
print(f"  - green:                {total_h2_green:,.2f} kg")
print(f"  - grey:                 {total_h2_grey:,.2f} kg")
print(f"H2 revenue (weighted):   EUR {h2_revenue:,.2f}")
print(f"Grid revenue (net):      EUR {grid_revenue:,.2f}")
print(f"Heat revenue (net):      EUR {heat_revenue:,.2f}")
print(f"Degradation cost:        EUR {degradation_cost:,.2f}")
print(f"NET TOTAL:               EUR {net_total:,.2f}")
print("====================\n")