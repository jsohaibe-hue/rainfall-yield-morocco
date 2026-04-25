"""
temporal_analysis.py
LBRTI2101A — Analyse des données temporelles
Séries temporelles, tendances, saisonnalité, corrélations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─── Palette cohérente ────────────────────────────────────────────────────────
COLORS = {
    "Casablanca-Settat": "#378ADD",
    "Fès-Meknès":        "#1D9E75",
    "Marrakech-Safi":    "#D85A30",
    "Rabat-Salé":        "#7F77DD",
    "Souss-Massa":       "#BA7517",
    "Oriental":          "#D4537E",
}
STYLE = {"axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"}
plt.rcParams.update(STYLE)

monthly = pd.read_csv("data/monthly_rainfall.csv")
annual  = pd.read_csv("data/annual_data.csv")

# ─── Figure 1 : Évolution annuelle par région ─────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

for region, color in COLORS.items():
    sub = annual[annual["region"] == region].sort_values("year")
    ax.plot(sub["year"], sub["annual_rainfall_mm"], color=color,
            linewidth=1.8, label=region, alpha=0.85)
    # Tendance linéaire
    z = np.polyfit(sub["year"], sub["annual_rainfall_mm"], 1)
    p = np.poly1d(z)
    ax.plot(sub["year"], p(sub["year"]), color=color,
            linewidth=1, linestyle="--", alpha=0.45)

ax.set_xlabel("Année", fontsize=11)
ax.set_ylabel("Précipitations annuelles (mm)", fontsize=11)
ax.set_title("Évolution des précipitations annuelles par région (2000–2024)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.5)
plt.tight_layout()
plt.savefig("outputs/01_annual_rainfall_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 1 : Tendances annuelles")

# ─── Figure 2 : Saisonnalité moyenne ─────────────────────────────────────────
month_names = ["Jan","Fév","Mar","Avr","Mai","Jui","Jul","Aoû","Sep","Oct","Nov","Déc"]
seasonal = monthly.groupby(["region","month"])["rainfall_mm"].mean().reset_index()

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
axes = axes.flatten()

for i, (region, color) in enumerate(COLORS.items()):
    sub = seasonal[seasonal["region"] == region].sort_values("month")
    axes[i].bar(range(1, 13), sub["rainfall_mm"], color=color, alpha=0.75, edgecolor="white")
    axes[i].set_title(region, fontsize=10, fontweight="bold")
    axes[i].set_xticks(range(1, 13))
    axes[i].set_xticklabels(month_names, rotation=45, fontsize=8)
    axes[i].set_ylabel("mm / mois", fontsize=9)

fig.suptitle("Saisonnalité des précipitations par région", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/02_seasonality_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 2 : Saisonnalité mensuelle")

# ─── Figure 3 : Série temporelle mensuelle (toutes régions cumulées) ──────────
fig, ax = plt.subplots(figsize=(14, 4))
monthly_total = monthly.groupby(["year","month"])["rainfall_mm"].mean().reset_index()
monthly_total["date_num"] = monthly_total["year"] + (monthly_total["month"] - 1) / 12

ax.fill_between(monthly_total["date_num"], monthly_total["rainfall_mm"],
                alpha=0.35, color="#378ADD")
ax.plot(monthly_total["date_num"], monthly_total["rainfall_mm"],
        color="#185FA5", linewidth=0.8)

# Moyenne mobile 12 mois
rolling = monthly_total.set_index("date_num")["rainfall_mm"].rolling(12, center=True).mean()
ax.plot(rolling.index, rolling.values, color="#D85A30", linewidth=2,
        label="Moyenne mobile 12 mois")

ax.set_xlabel("Année", fontsize=11)
ax.set_ylabel("Précipitations moyennes (mm)", fontsize=11)
ax.set_title("Série temporelle mensuelle — moyenne nationale (2000–2024)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/03_monthly_time_series.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 3 : Série temporelle mensuelle")

# ─── Figure 4 : Corrélation Pluviométrie ↔ Rendement ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, (region, color) in enumerate(COLORS.items()):
    sub = annual[annual["region"] == region]
    x = sub["annual_rainfall_mm"]
    y = sub["yield_qha"]

    axes[i].scatter(x, y, color=color, alpha=0.65, s=50, edgecolors="white", linewidths=0.5)

    # Régression linéaire
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xline = np.linspace(x.min(), x.max(), 100)
    axes[i].plot(xline, p(xline), color=color, linewidth=2, alpha=0.8)

    corr = np.corrcoef(x, y)[0, 1]
    axes[i].set_title(f"{region}\nr = {corr:.2f}", fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Précipitations (mm)", fontsize=9)
    axes[i].set_ylabel("Rendement (q/ha)", fontsize=9)

fig.suptitle("Corrélation précipitations annuelles ↔ rendement agricole", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_rainfall_yield_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 4 : Corrélations pluviométrie-rendement")

# ─── Figure 5 : Boxplot inter-annuel (variabilité) ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data_box = [annual[annual["region"] == r]["annual_rainfall_mm"].values for r in COLORS]
bp = ax.boxplot(data_box, patch_artist=True, notch=True,
                medianprops=dict(color="white", linewidth=2))

for patch, color in zip(bp["boxes"], COLORS.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks(range(1, len(COLORS)+1))
ax.set_xticklabels(list(COLORS.keys()), rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Précipitations annuelles (mm)", fontsize=11)
ax.set_title("Variabilité inter-annuelle des précipitations par région", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/05_rainfall_variability_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 5 : Variabilité inter-annuelle")

print("\n📊 Analyse temporelle terminée — 5 figures dans outputs/")
