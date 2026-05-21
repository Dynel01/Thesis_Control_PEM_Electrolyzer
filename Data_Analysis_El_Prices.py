import requests
import pandas as pd
import os
import time

# Official Elspotprices endpoint
API_URL = "https://api.energidataservice.dk/dataset/Elspotprices" 

# Directory Setup
RAW_DIR = "all_prices_hourly_DK"
NEG_DIR = "negative_prices_only_DK"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

YEARS = range(2013, 2026)
AREAS = ['DK1', 'DK2']
summary_stats = []

for year in YEARS:
    year_frames = []
    print(f"--- Processing {year} ---")
    
    for area in AREAS:
        params = {
            "start": f"{year}-01-01T00:00",
            "end": f"{year}-12-31T23:59",
            "filter": f'{{"PriceArea":["{area}"]}}',
            "limit": 10000 
        }
        
        try:
            r = requests.get(API_URL, params=params)
            r.raise_for_status()
            records = r.json().get('records', [])
            
            if not records:
                continue
                
            df_area = pd.DataFrame(records)
            
            # Ensure we are using the EUR column for standardized research
            # Typically 'SpotPriceEUR' in this dataset
            eur_col = 'SpotPriceEUR' if 'SpotPriceEUR' in df_area.columns else 'SpotPriceDKK'
            
            # Calculate stats
            neg_hours = len(df_area[df_area[eur_col] < 0])
            min_eur = df_area[eur_col].min()
            max_eur = df_area[eur_col].max()
            
            summary_stats.append({
                'Year': year, 
                'Area': area, 
                'Neg_Hours': neg_hours,
                'Min_Price_EUR': min_eur,
                'Max_Price_EUR': max_eur
            })
            
            year_frames.append(df_area)
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Error on {year} {area}: {e}")

    # Save Full and Negative CSVs
    if year_frames:
        df_full = pd.concat(year_frames)
        df_full.to_csv(f"{RAW_DIR}/full_hourly_prices_{year}.csv", index=False)
        
        # Determine the price column for filtering
        price_col = 'SpotPriceEUR' if 'SpotPriceEUR' in df_full.columns else 'SpotPriceDKK'
        df_full[df_full[price_col] < 0].to_csv(f"{NEG_DIR}/neg_prices_{year}.csv", index=False)

# Final Summary Table for PPT
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv("historical_price_summary_eur.csv", index=False)

print("\n--- FINAL HISTORICAL SUMMARY (EUR/MWh) ---")
print(summary_df)