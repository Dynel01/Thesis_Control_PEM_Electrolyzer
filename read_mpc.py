import pandas as pd
import matplotlib.pyplot as plt

plant_capacity_kw = 135000.0
min_operating_kw = 0.10 * plant_capacity_kw  # matches P_COLD / ON threshold

mpc_results = pd.read_csv("mpc_results_2.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(mpc_results['time'], mpc_results['P_stack'], label='Stack Power (kW)', color='blue')
ax1.plot(mpc_results['time'], mpc_results['wind'], label='Wind Available (kW)', color='green')
ax1.axhline(plant_capacity_kw, color='green', linestyle='--', label='Max Plant Capacity (kW)')
ax1.axhline(min_operating_kw, color='red', linestyle='--', label='Min Operating Capacity (kW)')
ax1.set_ylabel('Power (kW)')
ax1.legend(loc='upper right')

temp_c = mpc_results['T_stack'] - 273.15
ax2.plot(mpc_results['time'], temp_c, color='red', label='Stack Temp (°C)')
on_mask = mpc_results['y_on'] == 1
cold_start_mask = mpc_results['y_cold_start'] == 1
standby_mask = mpc_results['y_standby'] == 1

ax2.fill_between(
    mpc_results['time'], 0, 100, where=cold_start_mask,
    color='blue', alpha=0.2, label='Cold Start', step='mid'
)
ax2.fill_between(
    mpc_results['time'], 0, 100, where=standby_mask,
    color='yellow', alpha=0.2, label='Standby', step='mid'
)
ax2.fill_between(
    mpc_results['time'], 0, 100, where=on_mask,
    color='green', alpha=0.2, label='ON State', step='mid'
)
ax2.set_ylabel('Temp (°C)')
ax2.set_xlabel('Time Step (10 min)')
ax2.set_ylim(0, 100)
ax2.axhline(40, color='black', linestyle='--', linewidth=1, label='40°C')
ax2.axhline(80, color='black', linestyle='--', linewidth=1, label='80°C')

ax2.legend(loc='lower right')

plt.tight_layout()
# plt.savefig("mpc_results_plot.png", dpi=150)
print("Plot saved to mpc_results_plot.png")
plt.show()