"""
spatial_analysis.py
LBRTI2101A — Analyse des données spatiales
Carte choroplèthe, gradient spatial, interpolation IDW
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white"
})

annual  = pd.read_csv("data/annual_data.csv")
monthly = pd.read_csv("data/monthly_rainfall.csv")

# ─── Coordonnées approximatives des régions (centroïdes) ─────────────────────
REGION_COORDS = {
    "Casablanca-Settat": (33.59, -7.62),
    "Fès-Meknès":        (33.99, -5.00),
    "Marrakech-Safi":    (31.63, -8.01),
    "Rabat-Salé":        (34.01, -6.85),
    "Souss-Massa":       (30.42, -9.60),
    "Oriental":          (34.68, -1.90),
}

COLORS_MAP = {
    "Casablanca-Settat": "#378ADD",
    "Fès-Meknès":        "#1D9E75",
    "Marrakech-Safi":    "#D85A30",
    "Rabat-Salé":        "#7F77DD",
    "Souss-Massa":       "#BA7517",
    "Oriental":          "#D4537E",
}

# ─── Figure 6 : Carte bubble plot (précipitations moyennes par région) ────────
mean_annual = annual.groupby("region")["annual_rainfall_mm"].mean().reset_index()
mean_annual.columns = ["region", "mean_rain"]

fig, ax = plt.subplots(figsize=(9, 8))
ax.set_facecolor("#EAF3F8")

# Contour simplifié du Maroc (polygone approximatif)
maroc_lat = [35.9, 35.8, 35.1, 34.2, 33.0, 31.8, 30.0, 28.5, 27.7, 27.7,
             28.9, 29.5, 30.5, 31.5, 32.5, 33.5, 34.0, 34.8, 35.9]
maroc_lon = [-5.5, -2.2, -1.7, -1.1, -1.8, -1.3, -0.5, -0.5, -8.7, -13.2,
             -13.2, -13.1, -11.0, -9.8, -9.3, -8.4, -8.0, -6.2, -5.5]
ax.fill(maroc_lon, maroc_lat, color="#F5F0E8", zorder=1, alpha=0.9)
ax.plot(maroc_lon, maroc_lat, color="#AAAAAA", linewidth=1, zorder=2)

# Mer / frontières simplifiées
ax.set_xlim(-14.0, 0.0)
ax.set_ylim(27.0, 36.5)

# Colormap pour les bulles
cmap = plt.get_cmap("YlGnBu")
rain_vals = mean_annual.set_index("region")["mean_rain"]
norm = mcolors.Normalize(vmin=rain_vals.min() - 50, vmax=rain_vals.max() + 50)

for region, (lat, lon) in REGION_COORDS.items():
    rain = rain_vals.get(region, 300)
    size = (rain / rain_vals.max()) * 1800 + 200
    color = cmap(norm(rain))

    ax.scatter(lon, lat, s=size, color=color, alpha=0.85,
               edgecolors="#555555", linewidths=1.2, zorder=4)
    ax.annotate(
        f"{region}\n{rain:.0f} mm",
        xy=(lon, lat),
        xytext=(lon + 0.3, lat + 0.35),
        fontsize=8.5, fontweight="bold", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        zorder=5
    )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label("Précipitations moyennes annuelles (mm)", fontsize=9)

ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Carte des précipitations moyennes annuelles — Maroc (2000–2024)",
             fontsize=12, fontweight="bold", pad=12)
ax.grid(alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("outputs/06_spatial_rainfall_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 6 : Carte spatiale des précipitations")

# ─── Figure 7 : Interpolation IDW simplifiée (grille régulière) ──────────────
lats = np.array([v[0] for v in REGION_COORDS.values()])
lons = np.array([v[1] for v in REGION_COORDS.values()])
rains = np.array([rain_vals.get(r, 300) for r in REGION_COORDS])

lat_grid = np.linspace(27.5, 36.0, 120)
lon_grid = np.linspace(-13.5, 0.0, 160)
lon_g, lat_g = np.meshgrid(lon_grid, lat_grid)

# Calcul IDW (Inverse Distance Weighting)
grid_rain = np.zeros_like(lon_g)
power = 2
for i in range(len(lats)):
    dist = np.sqrt((lat_g - lats[i])**2 + (lon_g - lons[i])**2) + 1e-6
    grid_rain += rains[i] / dist**power

weights_sum = np.zeros_like(lon_g)
for i in range(len(lats)):
    dist = np.sqrt((lat_g - lats[i])**2 + (lon_g - lons[i])**2) + 1e-6
    weights_sum += 1 / dist**power

grid_rain /= weights_sum

fig, ax = plt.subplots(figsize=(10, 8))
cf = ax.contourf(lon_g, lat_g, grid_rain, levels=20, cmap="YlGnBu", alpha=0.85)
ax.contour(lon_g, lat_g, grid_rain, levels=8, colors="white", linewidths=0.5, alpha=0.4)

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label("Précipitations interpolées (mm/an)", fontsize=10)

# Points observés
sc = ax.scatter(lons, lats, c=rains, cmap="YlGnBu",
                s=120, edgecolors="black", linewidths=1.5, zorder=5)
for region, (lat, lon) in REGION_COORDS.items():
    ax.annotate(region.split("-")[0], xy=(lon, lat),
                xytext=(lon + 0.3, lat + 0.2), fontsize=8,
                fontweight="bold", color="black",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

ax.set_xlim(-13.5, 0.0)
ax.set_ylim(27.5, 36.0)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Interpolation spatiale IDW — précipitations annuelles (Maroc)",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("outputs/07_idw_interpolation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 7 : Interpolation IDW")

# ─── Figure 8 : Gradient Nord-Sud des précipitations ────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Scatter latitude vs pluie
for region, (lat, lon) in REGION_COORDS.items():
    rain = rain_vals.get(region, 300)
    ax1.scatter(lat, rain, color=COLORS_MAP[region], s=180,
                edgecolors="white", linewidths=1.5, zorder=4, label=region)
    ax1.annotate(region.split("-")[0], xy=(lat, rain),
                 xytext=(lat - 0.05, rain + 10), fontsize=8)

z = np.polyfit(lats, rains, 1)
p = np.poly1d(z)
xfit = np.linspace(lats.min() - 0.5, lats.max() + 0.5, 100)
ax1.plot(xfit, p(xfit), color="#555555", linewidth=1.5, linestyle="--", alpha=0.6)

ax1.set_xlabel("Latitude (°N)", fontsize=11)
ax1.set_ylabel("Précipitations moyennes (mm/an)", fontsize=11)
ax1.set_title("Gradient latitudinal des précipitations", fontsize=12, fontweight="bold")
ax1.legend(fontsize=7, loc="upper left")
ax1.grid(alpha=0.25)

# Barres horizontales comparatives
regions_sorted = mean_annual.sort_values("mean_rain", ascending=True)
bar_colors = [COLORS_MAP[r] for r in regions_sorted["region"]]
bars = ax2.barh(regions_sorted["region"], regions_sorted["mean_rain"],
                color=bar_colors, alpha=0.8, edgecolor="white", height=0.6)

for bar, val in zip(bars, regions_sorted["mean_rain"]):
    ax2.text(val + 5, bar.get_y() + bar.get_height()/2,
             f"{val:.0f} mm", va="center", fontsize=9, fontweight="bold")

ax2.set_xlabel("Précipitations moyennes (mm/an)", fontsize=11)
ax2.set_title("Classement des régions par pluviométrie", fontsize=12, fontweight="bold")
ax2.grid(alpha=0.25, axis="x")

plt.tight_layout()
plt.savefig("outputs/08_latitudinal_gradient.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 8 : Gradient spatial Nord-Sud")

print("\n🗺  Analyse spatiale terminée — 3 figures dans outputs/")
