"""
generate_data.py
Génère un dataset simulé mais réaliste de pluviométrie et rendements agricoles
pour 6 régions du Maroc sur 25 ans (2000-2024).
"""

import pandas as pd
import numpy as np

np.random.seed(42)

regions = {
    "Casablanca-Settat": {"lat": 33.59, "lon": -7.62, "base_rain": 380, "base_yield": 28},
    "Fès-Meknès":        {"lat": 33.99, "lon": -5.00, "base_rain": 520, "base_yield": 35},
    "Marrakech-Safi":    {"lat": 31.63, "lon": -8.01, "base_rain": 240, "base_yield": 20},
    "Rabat-Salé":        {"lat": 34.01, "lon": -6.85, "base_rain": 540, "base_yield": 32},
    "Souss-Massa":       {"lat": 30.42, "lon": -9.60, "base_rain": 180, "base_yield": 22},
    "Oriental":          {"lat": 34.68, "lon": -1.90, "base_rain": 310, "base_yield": 25},
}

records = []

for region, params in regions.items():
    for year in range(2000, 2025):
        # Tendance légère à la baisse (changement climatique simulé)
        trend_factor = 1 - 0.005 * (year - 2000)

        # Cycle multi-annuel (El Niño-like, période ~7 ans)
        cycle = 0.12 * np.sin(2 * np.pi * (year - 2000) / 7)

        for month in range(1, 13):
            # Saisonnalité : pluies concentrées nov-mars (méditerranéen/semi-aride)
            seasonal = (
                1.8 if month in [11, 12, 1, 2] else
                1.2 if month in [3, 10] else
                0.3 if month in [4, 9] else
                0.05
            )

            base = params["base_rain"] / 12
            noise = np.random.lognormal(0, 0.45)
            rainfall = max(0, base * seasonal * trend_factor * (1 + cycle) * noise)

            # Température (°C) — varie selon saison et latitude
            temp_base = 18 - (params["lat"] - 30) * 0.7
            temp_seasonal = -8 * np.cos(2 * np.pi * month / 12)
            temp = temp_base + temp_seasonal + np.random.normal(0, 0.8)

            records.append({
                "region": region,
                "latitude": params["lat"],
                "longitude": params["lon"],
                "year": year,
                "month": month,
                "rainfall_mm": round(rainfall, 2),
                "temperature_C": round(temp, 2),
            })

df_monthly = pd.DataFrame(records)

# Rendement annuel (lié à la pluviométrie annuelle, avec bruit)
annual = df_monthly.groupby(["region", "year"]).agg(
    annual_rainfall_mm=("rainfall_mm", "sum"),
    mean_temp_C=("temperature_C", "mean"),
    latitude=("latitude", "first"),
    longitude=("longitude", "first"),
).reset_index()

for region, params in regions.items():
    mask = annual["region"] == region
    rain_vals = annual.loc[mask, "annual_rainfall_mm"]
    rain_norm = (rain_vals - rain_vals.mean()) / rain_vals.std()

    base_y = params["base_yield"]
    noise = np.random.normal(0, 1.5, mask.sum())
    trend = -0.08 * (annual.loc[mask, "year"] - 2000)  # légère baisse tendancielle

    annual.loc[mask, "yield_qha"] = (
        base_y + 3.2 * rain_norm + trend + noise
    ).clip(lower=5).round(2)

annual["yield_qha"] = annual["yield_qha"].round(2)

df_monthly.to_csv("data/monthly_rainfall.csv", index=False)
annual.to_csv("data/annual_data.csv", index=False)

print(f"✅ Dataset mensuel  : {len(df_monthly)} lignes → data/monthly_rainfall.csv")
print(f"✅ Dataset annuel   : {len(annual)} lignes  → data/annual_data.csv")
print(f"\nAperçu données annuelles :")
print(annual.head(8).to_string(index=False))
