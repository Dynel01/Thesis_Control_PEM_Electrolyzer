import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#1. File paths
# ANH_FILES = [
#     r"Østed data\Østed data\RDdatasharing_ANH\Ørsted Confidential Information - SCADA_ANH_2013_JAN_JUN.csv",
#     r"Østed data\Østed data\RDdatasharing_ANH\Ørsted Confidential Information - SCADA_ANH_2013_JUL_DEC.csv",
#     r"Østed data\Østed data\RDdatasharing_ANH\Ørsted Confidential Information - SCADA_ANH_2014_JAN_JUN.csv",
#     r"Østed data\Østed data\RDdatasharing_ANH\Ørsted Confidential Information - SCADA_ANH_2014_JUL_DEC.csv"
# ]

# # Columns we need for Power Curves, Degradation, and Curtailment
# COLS = [
#     'TimeStamp', 'StationName', 'wtc_ActPower_mean', 
#     'wtc_AcWindSp_mean', 'wtc_PowerRef_endvalue', 'wtc_ActPower_stddev'
# ]

# print("Loading data...")
# raw_list = [pd.read_csv(f, usecols=COLS, parse_dates=['TimeStamp']) for f in ANH_FILES]
# df_raw = pd.concat(raw_list, ignore_index=True)

# # --- FILE 1: TURBINE-LEVEL (For Power Curves & Detailed Stats) ---
# # We keep StationName so you can plot individual turbine performance
# print("Saving Turbine-Level Master File...")
# df_raw.to_csv("Anholt_Turbine_Level_Master.csv", index=False)


# # --- FILE 2: HUB-LEVEL (For Electrolyzer Sizing & Financials) ---
# print("Aggregating to Hub-Level...")
# anh_hub = df_raw.groupby('TimeStamp').agg({
#     'wtc_ActPower_mean': 'sum',               # Total farm output
#     'wtc_PowerRef_endvalue': 'sum',     # Total farm reference (Curtailment)
#     'wtc_AcWindSp_mean': 'mean',        # Average farm wind speed
#     'wtc_ActPower_stddev': 'mean'       # Average fluctuation (Degradation proxy)
# }).reset_index()

# # Rename for downstream model logic
# anh_hub.rename(columns={
#     'wtc_ActPower_mean': 'wtc_ActPower_mean',
#     'wtc_PowerRef_endvalue': 'wtc_PowerRef_sum',
#     'wtc_ActPower_stddev': 'wtc_ActPower_std_avg'
# }, inplace=True)

# print("Saving Hub-Level Master File...")
# anh_hub.to_csv("Anholt_Hub_Level_Master.csv", index=False)

# print("Process Complete. You now have the raw turbine data and the aggregated farm data.")


"""""
Anholt Analysis
"""

# 1. SETUP & LOAD
# Use the file you generated: "Anholt_Turbine_Level_Master.csv"
file_name = "Anholt_Turbine_Level_Master.csv"
df = pd.read_csv(file_name, skip_blank_lines=True)

# Anholt Column Names
time_col = 'TimeStamp'
power_col = 'wtc_ActPower_mean'
wind_col = 'wtc_AcWindSp_mean'
ref_col = 'wtc_PowerRef_endvalue'
id_col = 'StationName'
std_col = 'wtc_ActPower_stddev'

df[time_col] = pd.to_datetime(df[time_col])
df.set_index(time_col, inplace=True)

# Filter for 2014 only as discussed
df = df[df.index.year == 2014]

print(f"Analysis Period: {df.index.min()} to {df.index.max()}")
print(f"Total Turbines: {df[id_col].nunique()}")

# 2. DATA CLEANING
# Anholt turbines: Cut-in ~4m/s, Rated ~12-13m/s, Cut-out 25m/s
mask = (
    (df[wind_col] > 0) & 
    (df[wind_col] < 25) & 
    ~((df[wind_col] > 3.5) & (df[power_col] <= 0))
)
df_clean = df[mask].copy()
df_clean.loc[df_clean[power_col] < 0, power_col] = 0

# # 3. GENERATING THE POWER CURVE (Figure 1)
# plt.figure(figsize=(12, 7))
# plt.scatter(df_clean[wind_col], df_clean[power_col], 
#             alpha=0.05, s=1, color='teal', label='Anholt SCADA Data')

# # Binned Average
# bins = np.arange(0, 26, 0.5)
# df_clean.loc[:, 'bin'] = pd.cut(df_clean[wind_col], bins)
# df_for_curve = df_clean[df_clean[ref_col] >= 3600]
# binned_avg = df_for_curve.groupby('bin', observed=False)[power_col].mean()

# plt.plot(bins[:-1], binned_avg.values, color='red', linewidth=3, label='Empirical Power Curve')

# plt.title('Anholt Offshore Wind Farm: Operational Power Curve (2014)', fontsize=14, fontweight='bold')
# plt.axvline(4, color='green', linestyle='--', label='Cut-in (4 m/s)')
# plt.axvline(13, color='orange', linestyle='--', label='Rated (13 m/s)')
# plt.xlabel('Wind Speed [m/s]')
# plt.ylabel('Active Power [kW]')
# plt.legend()
# plt.grid(True, alpha=0.3)
# # plt.savefig('Anholt_Power_Curve_2014.png', dpi=300)

# weights = np.ones(len(df_clean)) / len(df_clean) 

# # 4. WIND SPEED PROBABILITY DISTRIBUTION (Figure 2)
# plt.figure(figsize=(10, 6))
# plt.hist(df_clean[wind_col], bins=50, weights=weights, color='skyblue', edgecolor='white', alpha=0.7)
# plt.title('Wind Speed Relative Frequency Distribution (Anholt 2014)', fontsize=14)
# plt.xlabel('Wind Speed [m/s]')
# plt.ylabel('Probability Density')
# plt.grid(axis='y', alpha=0.3)
# # plt.savefig('Anholt_Wind_Distribution.png', dpi=300)

# # 5. POWER PROBABILITY DISTRIBUTION (Figure 3)
# plt.figure(figsize=(10, 6))
# plt.hist(df_clean[power_col], bins=50, weights=weights, color='goldenrod', edgecolor='white', alpha=0.7)
# plt.title('Power Output Relative Frequency Distribution (Anholt 2014)', fontsize=14)
# plt.xlabel('Active Power [kW]')
# plt.ylabel('Probability Density')
# plt.grid(axis='y', alpha=0.3)
# daily_wind = df_clean[wind_col].resample('D').mean()

# plt.figure(figsize=(15, 6))
# plt.plot(daily_wind.index, daily_wind, color='green', label='Daily Average Wind')
# plt.title('2014 Year Wind Speed Distribution')
# plt.ylabel('Wind Speed [m/s]')
# plt.grid(True, alpha=0.3)
# #plt.savefig('WMR_2Year_Wind_Distribution.png', dpi=300)
# plt.show()

# 3. GENERATING THE POWER CURVE (Figure 1)
# --- 1. WIND SPEED PROFILE (Figure A) ---
daily_wind = df_clean[wind_col].resample('D').mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily_wind.index, daily_wind, color='green', linewidth=1)
ax.set_ylabel('Wind Speed [m/s]', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_position([0.10, 0.15, 0.85, 0.75]) 

plt.savefig('Anholt_Hub/Anholt_Wind_Dist_Annual.png', dpi=300, bbox_inches='tight')
plt.close()


# --- 2. POWER CURVE (Figure B) ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(df_clean[wind_col], df_clean[power_col], 
           alpha=0.05, s=1, color='teal', label='Anholt SCADA Data')

bins = np.arange(0, 26, 0.5)
df_clean.loc[:, 'bin'] = pd.cut(df_clean[wind_col], bins)
df_for_curve = df_clean[df_clean[ref_col] >= 3600]
binned_avg = df_for_curve.groupby('bin', observed=False)[power_col].mean()

ax.plot(bins[:-1], binned_avg.values, color='red', linewidth=2.5, label='Empirical Power Curve')
ax.axvline(4, color='green', linestyle='--', label='Cut-in (4 m/s)')
ax.axvline(13, color='orange', linestyle='--', label='Rated (13 m/s)')

ax.set_xlabel('Wind Speed [m/s]', fontsize=11)
ax.set_ylabel('Active Power [kW]', fontsize=11)
ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
ax.grid(True, alpha=0.3)
ax.set_position([0.10, 0.15, 0.85, 0.75]) 

plt.savefig('Anholt_Hub/Anholt_Power_Curve.png', dpi=300, bbox_inches='tight')
plt.close()


weights = np.ones(len(df_clean)) / len(df_clean) 

# --- 3. WIND SPEED PROBABILITY DISTRIBUTION (Figure C) ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df_clean[wind_col], bins=50, weights=weights, color='skyblue', edgecolor='white', alpha=0.7)
ax.set_xlabel('Wind Speed [m/s]', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_position([0.10, 0.15, 0.85, 0.75]) 

plt.savefig('Anholt_Hub/Anholt_Wind_Freq.png', dpi=300, bbox_inches='tight')
plt.close()


# --- 4. POWER PROBABILITY DISTRIBUTION (Figure D) ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df_clean[power_col], bins=50, weights=weights, color='goldenrod', edgecolor='white', alpha=0.7)
ax.set_xlabel('Active Power [kW]', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle=':')
ax.set_position([0.10, 0.15, 0.85, 0.75]) 

plt.savefig('Anholt_Hub/Anholt_Power_Freq_Dist.png', dpi=300, bbox_inches='tight')
plt.close()
# 1. Calculate the time span
total_days = (df_clean.index.max() - df_clean.index.min()).days

# 2. Update to Anholt Turbine Count (111)
num_turbines_anholt = 111
expected_points = total_days * 144 * num_turbines_anholt 

actual_points = len(df_clean)
data_points = len(df)

# 3. Calculate Completeness
# Compares the cleaned data against a perfect 100% uptime scenario
data_completeness = (actual_points / expected_points) * 100

print(f"--- Anholt Data Coverage Summary (2014) ---")
print(f"Total Operational Days: {total_days}")
print(f"Number of Turbines:      {num_turbines_anholt}")
print(f"Expected SCADA Points:   {expected_points:,}")
print(f"Actual SCADA Points:     {actual_points:,}")
print(f"Data Completeness:       {data_completeness:.2f}%")

# 5. CAPACITY FACTOR & RESOURCE STATS
# Anholt: 111 turbines * 3600 kW = 399,600 kW
total_capacity_kw = 399600
aep_mwh = (df_clean[power_col].sum() * (10/60)) / 1000
theoretical_max_mwh = (total_capacity_kw * 8760)/1000

capacity_factor = (aep_mwh / theoretical_max_mwh) * 100

print(f"\n--- 2014 Resource Assessment ---")
print(f"Mean Wind Speed: {df_clean[wind_col].mean():.2f} m/s")
print(f"Annual Energy Production (AEP): {aep_mwh:,.2f} MWh")
print(f"Theoretical Max AEP: {theoretical_max_mwh:,.2f} MWh")
print(f"Calculated Capacity Factor: {capacity_factor:.2f} %")

# 6. CURTAILMENT ANALYSIS (PowerRef vs ActivePower)
# Rated capacity for Anholt Siemens 3.6MW is 3600 kW
rated_cap = 3600
curtailed_events = df[df[ref_col] < rated_cap]
avg_curt_hours = (len(curtailed_events) * (10/60)) / df[id_col].nunique()

print(f"Avg Curtailment per Turbine: {avg_curt_hours:.1f} hours/year")

# 7. 24-HOUR VARIABILITY PLOT (Most Variable Day)
most_variable_day = df_clean.groupby(df_clean.index.date)[wind_col].std().idxmax()
day_data = df_clean.loc[most_variable_day.strftime('%Y-%m-%d')]
# Sample first 144 points (24 hours) for a single turbine (e.g., A01)
sample_turbine = day_data[day_data[id_col] == 'A01'].iloc[0:144]
times = pd.date_range("00:00", "23:50", freq="10min")

plt.figure(figsize=(12, 5))
plt.plot(times, sample_turbine[wind_col], color='darkcyan', linewidth=2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.title(f'24-Hour Wind Profile - Turbine A01 ({most_variable_day})', fontweight='bold')
plt.ylabel('Wind Speed [m/s]')
plt.fill_between(times, sample_turbine[wind_col], color='darkcyan', alpha=0.1)
# plt.savefig('Anholt_24h_Variability.png', dpi=300)
plt.show()

# 11. POWER OUTPUT PROBABILITY DISTRIBUTION (FIXED Y-AXIS)
plt.figure(figsize=(12, 6))

# Create weights so the Y-axis represents the fraction of total time
weights = np.ones_like(df_clean[power_col]) / len(df_clean[power_col])

plt.hist(df_clean[power_col], bins=50, weights=weights, 
         color='goldenrod', alpha=0.7, edgecolor='white')

plt.title('Power Output Frequency Distribution: WMR (2 Years)', fontsize=14, fontweight='bold')
plt.xlabel('Active Power [kW]', fontsize=12)
plt.ylabel('Frequency (Fraction of Time)', fontsize=12) # This will now be 0.0 to 1.0
plt.grid(axis='y', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.show()

print(f"--- Thesis Resource Metrics ---")
# 12. SUMMARY STATISTICS
mean_ws = df_clean[wind_col].mean()
std_ws = df_clean[wind_col].std()
mean_pwr = df_clean[power_col].mean()
print(f"Mean Wind Speed: {mean_ws:.2f} m/s")
print(f"Wind Speed Std Dev: {std_ws:.2f} m/s")
print(f"Mean Turbine Power: {mean_pwr:.2f} kW")

# Maintenance: Turbine is consuming power (Active Power < 0)
# This usually indicates the turbine is "idling" or taking power from the grid for systems
# True Maintenance: Power is negative BUT the Grid allowed the turbine to run (Ref is high)
maintenance_data = df[(df[power_col] < 0) & (df[ref_col] >= 3600)]
maintenance_turbine_hours = len(maintenance_data) * (10 / 60)

# Curtailment: Turbine power reference (setpoint) is below rated capacity
# Rated capacity for Siemens SWT-3.6-120 at Anholt is 3600 kW
anh_rated_capacity = 3600
curtailment_data = df[df[ref_col] < anh_rated_capacity]
curtailment_turbine_hours = len(curtailment_data) * (10 / 60)

# Calculate averages per turbine
num_turbines = df['StationName'].nunique() # Should be 111
avg_maint_per_turbine = maintenance_turbine_hours / num_turbines
avg_curt_per_turbine = curtailment_turbine_hours / num_turbines

print(f"\n--- Operational Analysis (Anholt 2014 Full Year) ---")
print(f"Total Fleet Maintenance: {maintenance_turbine_hours:,.1f} turbine-hours")
print(f"Total Fleet Curtailment: {curtailment_turbine_hours:,.1f} turbine-hours")
print(f"---------------------------------------")
print(f"Avg. Maintenance per Turbine: {avg_maint_per_turbine:.1f} hours/year")
print(f"Avg. Curtailment per Turbine: {avg_curt_per_turbine:.1f} hours/year")

# --- 14. VISUALIZING CURTAILMENT EVENTS ---
if not curtailment_data.empty:
    # Resample to daily to show seasonal trends in curtailment
    curt_by_day = curtailment_data[ref_col].resample('D').count() * (10/60)
    plt.figure(figsize=(15, 5))
    plt.bar(curt_by_day.index, curt_by_day.values, color='salmon', label='Hours Throttled')
    plt.title('Daily Curtailment Hours: Anholt Fleet Total (2014)', fontsize=14)
    plt.ylabel('Aggregate Hours')
    plt.grid(axis='y', alpha=0.3)
    # plt.savefig('Anholt_Curtailment_Events_2014.png', dpi=300)
    plt.show()
    
# --- 15. DATA QUALITY & GAP ANALYSIS ---
null_counts = df[[wind_col, power_col]].isnull().sum()
print(f"\n--- Null Value Report ---")
null_pct = (null_counts / len(df)) * 100

print(f"\n--- Null Value Report ---")
for col in [wind_col, power_col]:
    print(f"{col}: {null_counts[col]} records ({null_pct[col]:.2f}%)")

# Time Gap Analysis: Checks for missing 10-min timestamps
time_diffs = df.index.to_series().diff().value_counts()
print(f"\n--- Time Gap Analysis (Expected: 00:10:00) ---")
print(time_diffs.head(5))

# --- DATA QUALITY: MISSING DAYS REPORT ---
# Create a full range of every single day in 2014
full_year_range = pd.date_range(start='2014-01-01', end='2014-12-31', freq='D')

# daily_wind comes from the df_clean resampling earlier
# We check which days in the calendar year are NOT in our dataset
missing_days = full_year_range[~full_year_range.isin(daily_wind.index.date)]

print(f"\n--- Missing Days Report ---")
if len(missing_days) > 0:
    print(f"Total days with ZERO data: {len(missing_days)}")
    print("Dates missing:")
    for d in missing_days:
        print(d.strftime('%Y-%m-%d'))
else:
    print("Every day in 2014 has at least one valid data point.")

# --- 16. OVER-RATING & PEAK ANALYSIS ---
# Max power for Anholt turbines (3.6 MW)
# Check for intervals exceeding 3.7 MW (roughly 2.5% over-rating)
max_pwr_clean = df_clean[power_col].max()
max_pwr_orig = df[power_col].max()
over_rated_limit = 3700 
over_rated_count = len(df_clean[df_clean[power_col] > over_rated_limit])
percent_over_rated = (over_rated_count / len(df_clean)) * 100

print(f"\n--- Power Peak Analysis ---")
print(f"Max Power (Cleaned): {max_pwr_clean:.2f} kW")
print(f"Max Power (Original): {max_pwr_orig:.2f} kW")
print(f"Intervals > 3.7 MW: {over_rated_count} ({percent_over_rated:.2f}%)")

# --- 17. STANDBY VS MAINTENANCE ---
# 1. Standby: Low wind (<4m/s), zero power
standby_mask = (df[wind_col] < 4) & (df[power_col] <= 0)
standby_hours = len(df[standby_mask]) * (10/60)

# 1. True Technical Fault (Wind is there, Ref is high, but Power is 0)
tech_fault_mask = (df[wind_col] >= 4) & (df[power_col] <= 0) & (df[ref_col] >= 3600)
tech_fault_hours = len(df[tech_fault_mask]) * (10/60)

# 2. Pure Curtailment (Ref is low, forcing Power to be low)
# This is the input power supply for the electrolyzer
curt_mask = (df[wind_col] >= 4) & (df[ref_col] < 3600)
curtailment_hours = len(df[curt_mask]) * (10/60)



print(f"\n--- System State Totals ---")
print(f"Total Standby Hours (Low Wind): {standby_hours:,.1f} turbine-hours")
print(f"Technical Fault Hours: {tech_fault_hours:,.1f} turbine-hours")
print(f"Curtailment Hours:     {curtailment_hours:,.1f} turbine-hours")

df_clean['ideal_power'] = df_clean['bin'].map(binned_avg).astype(float)

# 2. Curtailed Power is the difference (only when Ideal > Actual)
# We ignore negative values to avoid counting maintenance as 'curtailment'
df_clean['curtailed_kW'] = (df_clean['ideal_power'] - df_clean['wtc_ActPower_mean']).clip(lower=0)

# 3. Convert to Energy (MWh)
# Total = (sum of kW * 10/60 minutes) / 1000
total_curtailed_mwh = (df_clean['curtailed_kW'].sum() * (10/60)) / 1000
total_actual_mwh = (df_clean['wtc_ActPower_mean'].sum() * (10/60)) / 1000

print(f"--- Energy Yield Analysis ---")
print(f"Actual Energy Produced: {total_actual_mwh:,.2f} MWh")
print(f"Total Curtailed Energy: {total_curtailed_mwh:,.2f} MWh")
print(f"Curtailment Rate:       {(total_curtailed_mwh / (total_actual_mwh + total_curtailed_mwh)) * 100:.2f}%")

# --- 3. INTEGRATE ELECTRICITY PRICES ---
print("Processing 2014 Electricity Prices (DK1)...")
el_price_file_2014 = r"all_prices_hourly_DK\full_hourly_prices_2014.csv"
el_price = pd.read_csv(el_price_file_2014)

# Standardize timestamps
el_price['time_hourly'] = pd.to_datetime(el_price['HourDK'])

# Filter for DK1
dk1_prices = el_price[el_price['PriceArea'] == 'DK1'][['time_hourly', 'SpotPriceEUR']].copy()

# --- NEW: REMOVE DUPLICATE LABELS ---
# This fixes the "cannot reindex on an axis with duplicate labels" error
# We keep the first occurrence of any duplicate hour
dk1_prices = dk1_prices.drop_duplicates(subset=['time_hourly'], keep='first')

# RESAMPLING: Hourly -> 10 Minutes
# Uses '10min' resample alias ('10T' is deprecated in pandas)
dk1_prices_10min = dk1_prices.set_index('time_hourly').resample('10min').ffill().reset_index()
dk1_prices_10min.rename(columns={'time_hourly': 'TimeStamp', 'SpotPriceEUR': 'SpotPrice_DK1'}, inplace=True)
dk1_prices_10min['SpotPrice_DK1'] = dk1_prices_10min['SpotPrice_DK1']

hub_analysis = df_clean.groupby('TimeStamp').agg({
    'wtc_ActPower_mean': 'sum',
    'wtc_AcWindSp_mean': 'mean',
    'wtc_ActPower_stddev': lambda x: np.sqrt(np.sum(np.square(x))), # RSS for Hub Turbulence
    'wtc_PowerRef_endvalue': 'mean', # Hub-average setpoint
    'curtailed_kW': 'sum',
}).reset_index()

# 5. Peak Event Analysis
max_idx = hub_analysis['wtc_ActPower_mean'].idxmax()
min_idx = hub_analysis['wtc_ActPower_mean'].idxmin()

max_power = hub_analysis.loc[max_idx, 'wtc_ActPower_mean']
max_time = hub_analysis.loc[max_idx, 'TimeStamp']
min_power = hub_analysis.loc[min_idx, 'wtc_ActPower_mean']

# 6. Resource Metrics (For Thesis Summary)
total_actual_mwh = (hub_analysis['wtc_ActPower_mean'].sum() * (10/60)) / 1000
total_capacity_mw = total_capacity_kw/1000
total_hours = (hub_analysis['TimeStamp'].max() - hub_analysis['TimeStamp'].min()).total_seconds() / 3600
capacity_factor = (total_actual_mwh / (total_capacity_mw * total_hours)) * 100

# 7. Print Final Cleaned Statistics
print(f"--- 400 MW Hub Resource Assessment ---")
print(f"Dataset Range: {hub_analysis['TimeStamp'].min()} to {hub_analysis['TimeStamp'].max()}")
print(f"Cleaned Peak Power: {max_power:.2f} kW at {max_time}")
print(f"Cleaned Min Power:  {min_power:.2f} kW")


october_gap = df[
    (df.index >= '2014-10-01 00:00:00') & 
    (df.index <= '2014-10-07 00:00:00')
]

print(f"--- October Gap Diagnostic Report ---")
print(f"Found {len(october_gap)} raw turbine records during this window.")

if len(october_gap) == 0:
    print("RESULT: The days are completely missing from the raw SCADA file itself. This was a system-wide logging blackout.")
else:
    print("\nSample of raw data before the cleaning mask was applied:")
    # Key columns that triggered the mask
    sample_cols = [power_col, wind_col, ref_col, id_col, std_col]
    print(october_gap[sample_cols].head(10))
    
    print("\nAverage values during the gap:")
    print(f"Avg Wind Speed: {october_gap[wind_col].mean():.2f} m/s")
    print(f"Avg Power Output: {october_gap[power_col].mean():.2f} kW")
    
    
# 1. Reset index temporarily to access the 'TimeStamp' column
duplicates = df.reset_index().duplicated(subset=['TimeStamp', 'StationName'], keep=False)

# 2. Count the duplicates
num_duplicates = duplicates.sum()

print(f"--- Duplicate Check Report ---")
if num_duplicates > 0:
    print(f"Detected {num_duplicates} duplicate rows (rows with same Time and Turbine ID).")
    # Show the first few duplicates to see where they are coming from
    print(df[duplicates].sort_index().head(10))
else:
    print("No duplicates detected. File merge was clean.")
    
    
hub_master = pd.merge(hub_analysis, dk1_prices_10min, on='TimeStamp', how='inner')   
hub_master['Electricity Revenue'] = hub_master['wtc_ActPower_mean'] * hub_master['SpotPrice_DK1']/(1000*6)
hub_master['TI']= hub_master['wtc_ActPower_stddev'] / hub_master['wtc_ActPower_mean']
hub_master.to_csv('Anholt_hub_analysis.csv', index=False)

# 1. Constants & Thresholds
PEM_STACK_CAPACITY_KW = 13.5
SEC_H2 = 50                                  # kWh/kg
H2_PRICE_EUR = 5                             # €/kg
BREAKEVEN_POINT = (H2_PRICE_EUR * 1000) / SEC_H2 # 100 €/MWh
DISCOUNT_RATE = 0.08                         # 8%
CAPEX_PER_KW = 2196                          # €/kW
OPEX_PER_KW_YR = 54.9                        # €/kW/yr
STACK_REPLACE_PER_KW = 494                   # €/kW

# sizing_results = []


# for n in range(1000, 25100, 100):
#     PEM_CAPACITY_KW = n * PEM_STACK_CAPACITY_KW  
#     hub_master['PEM_Power_In_kW'] = 0.0
#     # Price < 100: Use wind
#     mask_below = hub_master['SpotPrice_DK1'] < BREAKEVEN_POINT
#     hub_master.loc[mask_below, 'PEM_Power_In_kW'] = hub_master['wtc_ActPower_mean'].clip(upper=PEM_CAPACITY_KW)
    
#     # Price < 0: Grid-Buy
#     mask_neg = hub_master['SpotPrice_DK1'] < 0
#     hub_master.loc[mask_neg, 'PEM_Power_In_kW'] = PEM_CAPACITY_KW
    
#     # Annualized Stats
#     h2_kg_total = (hub_master['PEM_Power_In_kW'] * (10/60) / SEC_H2).sum()
#     net_grid_flow = hub_master['wtc_ActPower_mean'] - hub_master['PEM_Power_In_kW']
#     elec_rev_total = (net_grid_flow * (10/60) * (hub_master['SpotPrice_DK1'] / 1000)).sum()
    
#     ann_h2_kg = h2_kg_total
#     ann_revenue = (ann_h2_kg * H2_PRICE_EUR + (elec_rev_total))
#     cap_factor = (hub_master['PEM_Power_In_kW'].mean() / PEM_CAPACITY_KW) * 100

#     # 3. Financial Modeling (20 Years)
#     total_capex = PEM_CAPACITY_KW * CAPEX_PER_KW
#     ann_opex_fixed = PEM_CAPACITY_KW * OPEX_PER_KW_YR
#     stack_replace_total = PEM_CAPACITY_KW * STACK_REPLACE_PER_KW
    
#     npv = -total_capex
#     pv_costs = total_capex
#     pv_h2 = 0
    
#     for yr in range(1, 21):
#         cf = ann_revenue - ann_opex_fixed
#         yr_cost = ann_opex_fixed
        
#         if yr in [8, 16]: # Replacements
#             cf -= stack_replace_total
#             yr_cost += stack_replace_total
            
#         npv += cf / (1 + DISCOUNT_RATE)**yr
#         pv_costs += yr_cost / (1 + DISCOUNT_RATE)**yr
#         pv_h2 += ann_h2_kg / (1 + DISCOUNT_RATE)**yr
        
#     lcoh = pv_costs / pv_h2 if pv_h2 > 0 else 0
    
#     # Store results
#     sizing_results.append({
#         'n': n,
#         'MW': PEM_CAPACITY_KW / 1000,
#         'NPV_MEUR': npv / 1e6,
#         'NPV/CAPEX': npv / total_capex,
#         'LCOH': lcoh,
#         'CapFactor': cap_factor,
#         'Annual_Revenue': ann_revenue
#     })

# # Convert to DataFrame to find the best result
# results_df = pd.DataFrame(sizing_results)
# # results_df['Score'] = (
# #     0.5 * (results_df['NPV_MEUR'] / results_df['NPV_MEUR'].max())
# #     - 0.3 * (results_df['LCOH'] / results_df['LCOH'].max())
# #     + 0.2 * (results_df['CapFactor'] / 100)
# # )
# ==============================================================================
# STANDALONE ELECTROLYZER SIZING LOOP
# ==============================================================================
# The wind farm revenue is removed. The electrolyzer is now treated as a 
# separate business entity that pays the wind farm the spot price for its energy.

sizing_results = []

for n in range(1000, 25100, 100):
    PEM_CAPACITY_KW = n * PEM_STACK_CAPACITY_KW  
    hub_master['PEM_Power_In_kW'] = 0.0
    
    # Rule 1: Price < Breakeven -> Use wind power up to capacity
    mask_below = hub_master['SpotPrice_DK1'] < BREAKEVEN_POINT
    hub_master.loc[mask_below, 'PEM_Power_In_kW'] = hub_master['wtc_ActPower_mean'].clip(upper=PEM_CAPACITY_KW)
    
    # Rule 2: Price < 0 -> Grid-Buy at maximum capacity
    mask_neg = hub_master['SpotPrice_DK1'] < 0
    hub_master.loc[mask_neg, 'PEM_Power_In_kW'] = PEM_CAPACITY_KW
    
    # Calculate physical hydrogen yield (kg)
    ann_h2_kg = (hub_master['PEM_Power_In_kW'] * (10/60) / SEC_H2).sum()
    
    # Calculate the exact cost of the electricity consumed by the electrolyzer
    # Energy (kWh) * Spot Price (€/kWh)
    elec_cost_total = (hub_master['PEM_Power_In_kW'] * (10/60) * (hub_master['SpotPrice_DK1'] / 1000)).sum()
    
    # Standalone Hydrogen Plant Revenue (Hydrogen Sales minus its Electricity Bill)
    h2_sales_revenue = ann_h2_kg * H2_PRICE_EUR
    ann_operational_profit = h2_sales_revenue - elec_cost_total
    
    # Electrolyzer Capacity Factor
    cap_factor = (hub_master['PEM_Power_In_kW'].mean() / PEM_CAPACITY_KW) * 100

    # Financial Parameters (Only applying to the Electrolyzer asset)
    total_capex = PEM_CAPACITY_KW * CAPEX_PER_KW
    ann_opex_fixed = PEM_CAPACITY_KW * OPEX_PER_KW_YR
    stack_replace_total = PEM_CAPACITY_KW * STACK_REPLACE_PER_KW
    
    # NPV initialized to the stand-alone chemical plant investment
    npv = -total_capex
    pv_costs = total_capex
    pv_h2 = 0
    
    # 20-Year Cash Flow Projection
    for yr in range(1, 21):
        # Net cash flow = Hydrogen Operational Profit - Fixed Maintenance OPEX
        cf = ann_operational_profit - ann_opex_fixed
        yr_cost = ann_opex_fixed
        
        # Account for periodic stack replacement cycles
        if yr in [8, 16]: 
            cf -= stack_replace_total
            yr_cost += stack_replace_total
            
        npv += cf / (1 + DISCOUNT_RATE)**yr
        pv_costs += yr_cost / (1 + DISCOUNT_RATE)**yr
        pv_h2 += ann_h2_kg / (1 + DISCOUNT_RATE)**yr
        
    # Calculate true stand-alone Levelized Cost of Hydrogen
    lcoh = (pv_costs + (elec_cost_total * sum(1 / (1 + DISCOUNT_RATE)**yr for yr in range(1, 21)))) / pv_h2 if pv_h2 > 0 else 0
    
    # Store clean standalone results
    sizing_results.append({
        'n': n,
        'MW': PEM_CAPACITY_KW / 1000,
        'NPV_MEUR': npv / 1e6,
        'NPV/CAPEX': npv / total_capex,
        'LCOH': lcoh,
        'CapFactor': cap_factor,
        'Annual_Operational_Profit_MEUR': ann_operational_profit / 1e6
    })

# Convert back to DataFrame for normal normalization and scoring
results_df = pd.DataFrame(sizing_results)
# ==============================================================================

# best_n = results_df.loc[results_df['Score'].idxmax()]
results_df['Annual_Operational_Profit_MEUR'] = results_df['Annual_Operational_Profit_MEUR'] / 1e6
# 1. Normalize NPV (Higher is better)
results_df['NPV_norm'] = (results_df['NPV_MEUR'] - results_df['NPV_MEUR'].min()) / \
                         (results_df['NPV_MEUR'].max() - results_df['NPV_MEUR'].min())

# 2. Normalize LCOH (Lower is better -> so we flip it)
# This makes the lowest price = 1.0 and the highest price = 0.0
results_df['LCOH_norm'] = (results_df['LCOH'].max() - results_df['LCOH']) / \
                          (results_df['LCOH'].max() - results_df['LCOH'].min())

# 3. Normalize Capacity Factor (Higher is better)
results_df['CF_norm'] = (results_df['CapFactor'] - results_df['CapFactor'].min()) / \
                         (results_df['CapFactor'].max() - results_df['CapFactor'].min())                        

# 4. Final Weighted Score
results_df['Score'] = (0.5 * results_df['NPV_norm']) + \
                     (0.3 * results_df['LCOH_norm']) + \
                     (0.2 * results_df['CF_norm'])

best_n = results_df.loc[results_df['Score'].idxmax()]
print(best_n)
                     
print(f"Sizing Optimization Complete.")
print(f"Optimal Stacks: {best_n['n']} ({best_n['MW']} MW)")
print(f"Max NPV: {best_n['NPV_MEUR']:.2f} Million €")
print(f"NPV/CAPEX: {best_n['NPV/CAPEX']:.2f}")
print(f"LCOH: {best_n['LCOH']:.2f} €/kg")
print(f"Electrolyzer Capacity Factor: {best_n['CapFactor']:.2f} %")
print(f"Optimal Annual Revenue: {best_n['Annual_Operational_Profit_MEUR']:.2f} Million €")


plt.figure(figsize=(10, 6))
plt.plot(results_df['MW'], results_df['LCOH'] )
plt.xlabel('PEM Plant Capacity(MW)')
plt.ylabel('LCOH [€/kg]')
plt.title('LCOH vs MW', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('LCOH_vs_MW_for_ANHOLT.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(results_df['MW'], results_df['CapFactor'])
plt.xlabel('PEM Plant Capacity(MW)')
plt.ylabel('Capacity Factor [%]')
plt.title('Capacity Factor vs MW', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('Capacity_Factor_vs_MW_for_ANHOLT.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(results_df['MW'], results_df['NPV/CAPEX'])
plt.xlabel('PEM Plant Capacity(MW)')
plt.ylabel('NPV/CAPEX')
plt.title('NPV/CAPEX vs MW', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('NPV_vs_CAPEX_vs_MW_for_ANHOLT.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(results_df['MW'], results_df['NPV_MEUR'])
plt.xlabel('PEM Plant Capacity(MW)')
plt.ylabel('NPV [Million €]')
plt.title('NPV vs MW', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('NPV_vs_MW_for_ANHOLT.png', dpi=300)

# --- STEP 6: MARKET CONTEXT & PRICE ANALYSIS ---

# 1. Aggregate to Hourly level for cleaner reporting
# We convert 10-min power [kW] to hourly energy [MWh]
hourly_impact = hub_master.groupby(hub_master['TimeStamp'].dt.floor('h')).agg({
    'wtc_ActPower_mean': lambda x: (x.sum() * (10/60)) / 1000,
    'SpotPrice_DK1': 'first',
    'wtc_AcWindSp_mean': 'mean',
    'TI': 'mean'
}).reset_index().rename(columns={'wtc_ActPower_mean': 'actual_MWh', 'TimeStamp': 'time_hourly'})

# Calculate Curtailed MWh at hourly level
# (Using the 2.97% rate or the specific curtailed_kW column if you have it in hub_master)
if 'curtailed_kW' in hub_master.columns:
    hourly_curtailment = hub_master.groupby(hub_master['TimeStamp'].dt.floor('h'))['curtailed_kW'].apply(lambda x: (x.sum() * (10/60)) / 1000)
    hourly_impact['curtailed_MWh'] = hourly_curtailment.values

# 2. Perspective A: Top Curtailment Hours (The Energy Opportunity)
top_curtailment = hourly_impact.sort_values(by='curtailed_MWh', ascending=False).head(10)

# 3. Perspective B: Negative Price Hours (The Economic Opportunity)
neg_price_hours = hourly_impact[hourly_impact['SpotPrice_DK1'] < 0]
total_neg_mwh = neg_price_hours['curtailed_MWh'].sum() if 'curtailed_MWh' in hourly_impact.columns else 0

print(f"\n--- MARKET ANALYSIS SUMMARY (DK1 2014) ---")
print(f"Total Hours with Negative Prices: {len(neg_price_hours)}")
print(f"Total Energy Curtailed during Negative Prices: {total_neg_mwh:.2f} MWh")

print("\n--- TOP 10 CURTAILMENT EVENTS (Potential H2 Peaks) ---")
print(top_curtailment[['time_hourly', 'curtailed_MWh', 'SpotPrice_DK1']].to_string(index=False))

# 4. Find 'Clogged Grid' Events 
# High curtailment despite positive prices (indicates local grid congestion)
anomaly = hourly_impact[(hourly_impact['SpotPrice_DK1'] > 20) & (hourly_impact['curtailed_MWh'] > 15)]
print("\n--- THE 'CLOGGED GRID' EVENTS (High Price + High Curtailment) ---")
print(anomaly[['time_hourly', 'curtailed_MWh', 'SpotPrice_DK1']])

# 5. Visualization: Correlation Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(hourly_impact['SpotPrice_DK1'], hourly_impact['curtailed_MWh'], 
            alpha=0.4, color='forestgreen', s=20)
plt.axvline(0, color='red', linestyle='--', label='Negative Price Threshold')
plt.title('Anholt 2014: Curtailment Volume vs. Spot Price', fontweight='bold', fontsize=14)
plt.xlabel('Spot Price [EUR/MWh]', fontsize=12)
plt.ylabel('Curtailed Energy [MWh]', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('Anholt_Market_Correlation.png', dpi=300)
plt.show()

clean_df = hourly_impact[(hourly_impact['wtc_AcWindSp_mean'] >= 3.0) & 
              (hourly_impact['TI'] > 0.0) & 
              (hourly_impact['TI'] <= 0.50)]

# 4. Plot the realistic data
plt.figure(figsize=(10, 6))
plt.scatter(clean_df['wtc_AcWindSp_mean'], clean_df['TI'], alpha=0.3, s=5, color='royalblue')
plt.title('Turbulence Intensity vs Wind Speed (2014 Cleaned SCADA)')
plt.xlabel('Mean Wind Speed [m/s]')
plt.ylabel('Turbulence Intensity [-]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 25)
plt.ylim(0, 0.5) # Keeps the chart locked into the 0% - 50% realistic window
# plt.savefig("Cleaned_Turbulence_Plot.png")
plt.show()
