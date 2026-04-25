"""
ml_prediction.py
LBRTI2101 — Machine Learning
Prédiction du rendement agricole à partir des données climatiques
Modèles : Régression linéaire, Random Forest, évaluation et interprétation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"
})

annual  = pd.read_csv("data/annual_data.csv")
monthly = pd.read_csv("data/monthly_rainfall.csv")

# ─── Feature engineering ─────────────────────────────────────────────────────
# Précipitations par saison
def get_season_rain(monthly, region, year):
    sub = monthly[(monthly["region"] == region) & (monthly["year"] == year)]
    winter = sub[sub["month"].isin([12, 1, 2])]["rainfall_mm"].sum()
    spring = sub[sub["month"].isin([3, 4, 5])]["rainfall_mm"].sum()
    summer = sub[sub["month"].isin([6, 7, 8])]["rainfall_mm"].sum()
    autumn = sub[sub["month"].isin([9, 10, 11])]["rainfall_mm"].sum()
    return winter, spring, summer, autumn

rows = []
for _, row in annual.iterrows():
    w, sp, su, au = get_season_rain(monthly, row["region"], row["year"])
    rows.append({"region": row["region"], "year": row["year"],
                 "rain_winter": w, "rain_spring": sp,
                 "rain_summer": su, "rain_autumn": au})

seasons_df = pd.DataFrame(rows)
df = annual.merge(seasons_df, on=["region", "year"])

# Encodage région
le = LabelEncoder()
df["region_enc"] = le.fit_transform(df["region"])

# Année normalisée
df["year_norm"] = (df["year"] - 2000) / 24

# Features finales
FEATURES = ["annual_rainfall_mm", "mean_temp_C", "rain_winter", "rain_spring",
            "rain_summer", "rain_autumn", "year_norm", "region_enc"]
FEAT_LABELS = ["Pluie annuelle", "Temp. moyenne", "Pluie hiver", "Pluie printemps",
               "Pluie été", "Pluie automne", "Tendance annuelle", "Région"]

X = df[FEATURES].values
y = df["yield_qha"].values

# ─── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Modèles ─────────────────────────────────────────────────────────────────
models = {
    "Régression linéaire": LinearRegression(),
    "Ridge Regression":    Ridge(alpha=1.0),
    "Random Forest":       RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {
        "model": model, "y_pred": y_pred,
        "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std(),
        "rmse": rmse, "mae": mae, "r2": r2
    }
    print(f"  {name:25s}  CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
          f"Test R²={r2:.3f}  RMSE={rmse:.2f}")

best_name = max(results, key=lambda k: results[k]["r2"])
best = results[best_name]
print(f"\n🏆 Meilleur modèle : {best_name} (R² = {best['r2']:.3f})")

# ─── Figure 9 : Comparaison des modèles ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

names = list(results.keys())
r2_vals = [results[n]["r2"] for n in names]
rmse_vals = [results[n]["rmse"] for n in names]

bar_colors = ["#378ADD" if n != best_name else "#1D9E75" for n in names]

bars = ax1.bar(names, r2_vals, color=bar_colors, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, r2_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("R² (test set)", fontsize=11)
ax1.set_title("Comparaison des modèles — R²", fontsize=12, fontweight="bold")
ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)

bars2 = ax2.bar(names, rmse_vals, color=bar_colors, alpha=0.85, edgecolor="white")
for bar, val in zip(bars2, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylabel("RMSE (q/ha)", fontsize=11)
ax2.set_title("Comparaison des modèles — RMSE", fontsize=12, fontweight="bold")
ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/09_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 9 : Comparaison des modèles")

# ─── Figure 10 : Prédictions vs Valeurs réelles ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    y_pred = res["y_pred"]
    ax.scatter(y_test, y_pred, alpha=0.6, color="#378ADD" if name != best_name else "#1D9E75",
               s=50, edgecolors="white", linewidths=0.5)

    lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="Parfait")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Rendement réel (q/ha)", fontsize=10)
    ax.set_ylabel("Rendement prédit (q/ha)", fontsize=10)
    ax.set_title(f"{name}\nR²={res['r2']:.3f}  RMSE={res['rmse']:.2f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

plt.suptitle("Prédictions vs Valeurs réelles — tous modèles", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/10_predictions_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 10 : Prédictions vs valeurs réelles")

# ─── Figure 11 : Feature Importance (Random Forest) ─────────────────────────
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#1D9E75" if i == indices[0] else "#B5D4F4" for i in range(len(FEATURES))]
bars = ax.barh([FEAT_LABELS[j] for j in indices[::-1]],
               importances[indices[::-1]],
               color=colors[::-1], alpha=0.85, edgecolor="white")

for bar, val in zip(bars, importances[indices[::-1]]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax.set_xlabel("Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_title("Importance des variables — Random Forest", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/11_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 11 : Importance des variables")

# ─── Figure 12 : Projection future (2025-2035) ───────────────────────────────
future_years = np.arange(2025, 2036)
regions_list = list(df["region"].unique())
predictions_future = []

for region in regions_list:
    sub = df[df["region"] == region].copy()
    mean_rain = sub["annual_rainfall_mm"].mean()
    mean_temp = sub["mean_temp_C"].mean()
    rain_trend = np.polyfit(sub["year"], sub["annual_rainfall_mm"], 1)[0]

    for yr in future_years:
        proj_rain = max(50, mean_rain + rain_trend * (yr - 2012))
        X_future = np.array([[proj_rain, mean_temp,
                               proj_rain * 0.45, proj_rain * 0.20,
                               proj_rain * 0.03, proj_rain * 0.32,
                               (yr - 2000) / 24,
                               le.transform([region])[0]]])
        pred_yield = rf_model.predict(X_future)[0]
        predictions_future.append({"region": region, "year": yr,
                                   "proj_rain": proj_rain, "pred_yield": pred_yield})

df_future = pd.DataFrame(predictions_future)

COLORS_MAP = {
    "Casablanca-Settat": "#378ADD", "Fès-Meknès": "#1D9E75",
    "Marrakech-Safi": "#D85A30", "Rabat-Salé": "#7F77DD",
    "Souss-Massa": "#BA7517", "Oriental": "#D4537E",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for region in regions_list:
    color = COLORS_MAP.get(region, "#888780")
    hist = df[df["region"] == region].sort_values("year")
    fut = df_future[df_future["region"] == region].sort_values("year")

    ax1.plot(hist["year"], hist["annual_rainfall_mm"], color=color, linewidth=1.5)
    ax1.plot(fut["year"], fut["proj_rain"], color=color,
             linewidth=1.5, linestyle="--", alpha=0.7)

    ax2.plot(hist["year"], hist["yield_qha"], color=color, linewidth=1.5, label=region)
    ax2.plot(fut["year"], fut["pred_yield"], color=color,
             linewidth=1.5, linestyle="--", alpha=0.7)

ax1.axvline(2025, color="#888780", linestyle=":", linewidth=1.2, alpha=0.7)
ax1.set_title("Projection des précipitations (2025–2035)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Année"); ax1.set_ylabel("Précipitations (mm)")

ax2.axvline(2025, color="#888780", linestyle=":", linewidth=1.2, alpha=0.7)
ax2.set_title("Projection du rendement agricole (2025–2035)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Année"); ax2.set_ylabel("Rendement (q/ha)")
ax2.legend(fontsize=7.5, loc="upper right")

for ax in [ax1, ax2]:
    ax.text(2025.1, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05,
            "Projection →", fontsize=8, color="#888780")
    ax.grid(alpha=0.25)

plt.suptitle("Projection climatique et impact agricole — Random Forest", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/12_future_projection.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Figure 12 : Projections 2025-2035")

# ─── Sauvegarde des métriques ─────────────────────────────────────────────────
metrics_df = pd.DataFrame([{
    "Modèle": name,
    "CV R² moyen": f"{res['cv_r2_mean']:.4f}",
    "CV R² std": f"{res['cv_r2_std']:.4f}",
    "Test R²": f"{res['r2']:.4f}",
    "RMSE (q/ha)": f"{res['rmse']:.3f}",
    "MAE (q/ha)": f"{res['mae']:.3f}",
} for name, res in results.items()])
metrics_df.to_csv("outputs/model_metrics.csv", index=False)
print("\n📊 Machine Learning terminé — 4 figures + métriques dans outputs/")
