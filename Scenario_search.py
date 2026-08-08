import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Anholt_hub_analysis.csv")
WINDOW_LEN = 432  # 3 days at 10-min resolution

candidate_starts = range(0, len(df) - WINDOW_LEN, 144)

stats = []
for start in candidate_starts:
    window_price = df['SpotPrice_DK1'].iloc[start:start+WINDOW_LEN] * 3
    window_wind = df['wtc_ActPower_mean'].iloc[start:start+WINDOW_LEN]
    stats.append({
        'start': start,
        'price_mean': window_price.mean(),
        'price_std': window_price.std(),
        'wind_mean': window_wind.mean(),
        'wind_std': window_wind.std(),
        'wind_max': window_wind.max()
    })

stats_df = pd.DataFrame(stats)

# --- Scenario 1: High volatility / high price ---
volatility_candidates = stats_df.sort_values('price_std', ascending=False).head(30)
volatility_best = volatility_candidates.sort_values('wind_std', ascending=False).iloc[0]
scenario1_idx = int(volatility_best['start'])
print("Scenario 1 (High volatility/high price) BASE_IDX:", scenario1_idx)
print("  price_mean:", round(volatility_best['price_mean'],1),
      "price_std:", round(volatility_best['price_std'],1),
      "wind_mean:", round(volatility_best['wind_mean']))

# --- Scenario 2: Scarcity -- genuinely low wind throughout, no near-capacity spikes ---
scarcity_candidates = stats_df.sort_values('wind_mean').head(20)
scarcity_candidates = scarcity_candidates[scarcity_candidates['wind_max'] < 200000]
scarcity_best = scarcity_candidates.sort_values('price_mean', ascending=False).iloc[0]
scenario2_idx = int(scarcity_best['start'])
print("\nScenario 2 (Scarcity) BASE_IDX:", scenario2_idx)
print("  wind_mean:", round(scarcity_best['wind_mean']), "wind_max:", round(scarcity_best['wind_max']),
      "price_mean:", round(scarcity_best['price_mean'],1))

# --- Scenario 3: Cannibalization -- high wind, low price ---
cannibal_candidates = stats_df.sort_values('wind_mean', ascending=False).head(30)
cannibal_best = cannibal_candidates.sort_values('price_mean').iloc[0]
scenario3_idx = int(cannibal_best['start'])
print("\nScenario 3 (Cannibalization) BASE_IDX:", scenario3_idx)
print("  price_mean:", round(cannibal_best['price_mean'],1),
      "wind_mean:", round(cannibal_best['wind_mean']))

# --- Scenario 4: Generic -- closest to dataset-wide median conditions ---
overall_price_median = (df['SpotPrice_DK1'] * 3).median()
overall_wind_median = df['wtc_ActPower_mean'].median()

# Normalize distance by each metric's own scale so price and wind
# contribute comparably to the "closeness" measure
price_range = stats_df['price_mean'].max() - stats_df['price_mean'].min()
wind_range = stats_df['wind_mean'].max() - stats_df['wind_mean'].min()

stats_df['distance_from_median'] = (
    ((stats_df['price_mean'] - overall_price_median).abs() / price_range) +
    ((stats_df['wind_mean'] - overall_wind_median).abs() / wind_range)
)

generic_best = stats_df.sort_values('distance_from_median').iloc[0]
scenario4_idx = int(generic_best['start'])
print("\nScenario 4 (Generic) BASE_IDX:", scenario4_idx)
print("  price_mean:", round(generic_best['price_mean'],1),
      "wind_mean:", round(generic_best['wind_mean']),
      "(dataset median price:", round(overall_price_median,1),
      ", dataset median wind:", round(overall_wind_median), ")")


# =========================================================
# FREQUENCY WEIGHTING -- classify every window in the year against
# the four chosen archetypes, to get real-world occurrence weights
# =========================================================

archetypes = {
    "Volatility": volatility_best,
    "Scarcity": scarcity_best,
    "Cannibalization": cannibal_best,
    "Generic": generic_best,
}

# Features used for classification distance -- same three used to select the archetypes
features = ['price_mean', 'price_std', 'wind_mean']

# Normalize each feature by its standard deviation across all windows in the year,
# so price and wind contribute comparably to the distance measure regardless of units/scale
feature_stds = {f: stats_df[f].std() for f in features}

def normalized_distance(row, archetype):
    return sum(
        ((row[f] - archetype[f]) / feature_stds[f]) ** 2
        for f in features
    ) ** 0.5

# For every window in the dataset, compute distance to each archetype and assign to the nearest
distances = pd.DataFrame({
    name: stats_df.apply(lambda row: normalized_distance(row, arch), axis=1)
    for name, arch in archetypes.items()
})
stats_df['nearest_scenario'] = distances.idxmin(axis=1)

# Frequency table: how many windows (and what fraction of the year) fall into each archetype
freq_counts = stats_df['nearest_scenario'].value_counts()
freq_fraction = freq_counts / len(stats_df)

# Convert window-count fraction into an approximate day-count per year,
# useful for annualized/LCOH-style extrapolation later
days_per_year_equivalent = freq_fraction * 365

print("\n" + "=" * 55)
print("SCENARIO FREQUENCY WEIGHTS (based on full-year classification)")
print("=" * 55)
for name in archetypes:
    count = freq_counts.get(name, 0)
    frac = freq_fraction.get(name, 0.0)
    days = days_per_year_equivalent.get(name, 0.0)
    print(f"  {name:18s}: {count:4d} windows  |  {frac*100:5.1f}% of year  |  ~{days:5.1f} days/year")
print("=" * 55)

# Save for downstream use (e.g. in the formulation-comparison / expected-value scripts)
freq_weights_df = pd.DataFrame({
    'scenario': list(archetypes.keys()),
    'fraction_of_year': [freq_fraction.get(name, 0.0) for name in archetypes],
    'approx_days_per_year': [days_per_year_equivalent.get(name, 0.0) for name in archetypes],
})
# freq_weights_df.to_csv('scenario_frequency_weights.csv', index=False)
print("\nSaved scenario_frequency_weights.csv")
# =========================================================
# PLOT -- uses the freshly computed indices, not hardcoded values
# =========================================================
scenarios = {
    "Scenario 1: High Volatility": scenario1_idx,
    "Scenario 2: Scarcity": scenario2_idx,
    "Scenario 3: Cannibalization": scenario3_idx,
    "Scenario 4: Generic": scenario4_idx
}

# Adjusted figure size and spacing layout
fig, axes = plt.subplots(len(scenarios), 2, figsize=(14, 12), sharex=False)

for row, (label, base_idx) in enumerate(scenarios.items()):
    price = df['SpotPrice_DK1'].iloc[base_idx:base_idx+WINDOW_LEN].values * 3
    wind = df['wtc_ActPower_mean'].iloc[base_idx:base_idx+WINDOW_LEN].values / 1000.0
    time_steps = range(WINDOW_LEN)

    # --- Electricity Price Subplot ---
    ax_price = axes[row, 0]
    ax_price.plot(time_steps, price, color='purple', linewidth=1.2)
    ax_price.set_title(f"{label} — Electricity Price", fontsize=11, pad=8)
    ax_price.set_ylabel("Price (EUR/MWh)", fontsize=9)
    ax_price.axhline(price.mean(), color='gray', linestyle='--', linewidth=1,
                      label=f"Mean: {price.mean():.1f}")
    
    # Large headroom so lines never reach the top-right legend zone
    p_max = max(price.max(), price.mean())
    ax_price.set_ylim(price.min() - (p_max - price.min()) * 0.05, p_max * 1.45)
    
    # Keep legend neatly inside the plot area, safely above the data lines
    ax_price.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax_price.grid(True, linestyle=':', alpha=0.5)

    # --- Wind Power Subplot ---
    ax_wind = axes[row, 1]
    ax_wind.plot(time_steps, wind, color='green', linewidth=1.2)
    ax_wind.axhline(135.0, color='red', linestyle='--', linewidth=1, label='Plant Capacity')
    ax_wind.set_title(f"{label} — Wind Power", fontsize=11, pad=8)
    ax_wind.set_ylabel("Wind Available (MW)", fontsize=9)
    ax_wind.axhline(wind.mean(), color='gray', linestyle='--', linewidth=1,
                     label=f"Mean: {wind.mean():.1f}")
    
    # Large headroom for wind
    w_max = max(wind.max(), 135.0, wind.mean())
    ax_wind.set_ylim(0, w_max * 1.45)
    
    # Keep legend neatly inside the plot area, safely above the data lines
    ax_wind.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax_wind.grid(True, linestyle=':', alpha=0.5)

# X-axis labels only on the bottom row to reduce clutter
axes[-1, 0].set_xlabel("Time Step (10-min intervals)", fontsize=10)
axes[-1, 1].set_xlabel("Time Step (10-min intervals)", fontsize=10)

# Ensures proper spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.subplots_adjust(hspace=0.45, wspace=0.2)

plt.savefig("scenario_comparison.png", dpi=300, bbox_inches='tight')
plt.show()