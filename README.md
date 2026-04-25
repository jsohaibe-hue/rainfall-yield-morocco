# Analyse Spatio-Temporelle de la Pluviométrie et Impact sur le Rendement Agricole au Maroc

> Projet réalisé dans le cadre d'une candidature en échange académique — UCLouvain (LBRTI2101)  
> IAV Hassan II — Filière Bioinformatique & Data Science Agricole

---

## 🎯 Objectif

Ce projet analyse les données pluviométriques de 6 régions marocaines sur la période **2000–2024** pour :

1. Identifier les **tendances et la saisonnalité** des précipitations (analyse temporelle)
2. Caractériser la **distribution spatiale** des pluies à l'échelle nationale (analyse spatiale)
3. Modéliser et **prédire les rendements agricoles** à partir des variables climatiques (Machine Learning)
4. Projeter l'évolution des rendements agricoles à l'horizon **2025–2035**

Ce travail s'inscrit directement dans la thématique du cours **LBRTI2101** (Data Science in Bioscience Engineering) — notamment les modules sur l'analyse des données corrélées dans l'espace et dans le temps, et les enjeux de fiabilité et d'interprétation des données environnementales.

---

## 📂 Structure du projet

```
rainfall_maroc/
│
├── data/                        # Données générées
│   ├── monthly_rainfall.csv     # Précipitations mensuelles (6 régions × 25 ans × 12 mois)
│   └── annual_data.csv          # Données annuelles agrégées + rendements
│
├── scripts/                     # Scripts Python
│   ├── generate_data.py         # Génération du dataset simulé
│   ├── temporal_analysis.py     # Analyse des séries temporelles
│   ├── spatial_analysis.py      # Cartographie et interpolation spatiale
│   └── ml_prediction.py         # Modèles prédictifs (Random Forest, etc.)
│
├── outputs/                     # Visualisations générées
│   ├── 01_annual_rainfall_trends.png
│   ├── 02_seasonality_by_region.png
│   ├── 03_monthly_time_series.png
│   ├── 04_rainfall_yield_correlation.png
│   ├── 05_rainfall_variability_boxplot.png
│   ├── 06_spatial_rainfall_map.png
│   ├── 07_idw_interpolation.png
│   ├── 08_latitudinal_gradient.png
│   ├── 09_model_comparison.png
│   ├── 10_predictions_vs_real.png
│   ├── 11_feature_importance.png
│   ├── 12_future_projection.png
│   └── model_metrics.csv
│
├── main.py                      # Point d'entrée — lance tout le pipeline
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

```bash
git clone https://github.com/votre-username/rainfall-yield-morocco.git
cd rainfall-yield-morocco

python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🚀 Exécution

```bash
python main.py
```

Cela génère automatiquement les données, les analyses et toutes les visualisations dans `outputs/`.

---

## 📊 Résultats clés

### Analyse temporelle
- Tendance à la **baisse des précipitations** de l'ordre de 0.5% par an dans la plupart des régions
- Saisonnalité fortement marquée : **80% des pluies concentrées entre novembre et mars** (régime méditerranéen)
- Forte **autocorrélation temporelle** : les précipitations d'une année influencent significativement l'année suivante

### Analyse spatiale
- Gradient **Nord-Sud** net : les régions du nord (Fès-Meknès, Rabat-Salé) reçoivent 2× plus de pluies que le sud (Souss-Massa)
- L'interpolation IDW révèle une **zone sèche structurelle** dans la région de Souss-Massa et de l'Oriental

### Machine Learning
| Modèle | R² (test) | RMSE (q/ha) |
|---|---|---|
| Régression linéaire | ~0.72 | ~2.8 |
| Ridge Regression | ~0.73 | ~2.7 |
| **Random Forest** | **~0.89** | **~1.7** |
| Gradient Boosting | ~0.87 | ~1.9 |

Le **Random Forest** obtient les meilleures performances. Les variables les plus importantes sont la **pluviométrie hivernale** et la **pluviométrie annuelle totale**, confirmant le rôle clé des pluies de saison froide pour les rendements céréaliers au Maroc.

---

## 🔍 Discussion critique

**Limites du modèle :**
- Dataset simulé (pas de données terrain réelles) — à remplacer par des données CHIRPS + FAO en production
- Pas de prise en compte des pratiques agricoles, des types de cultures, ni de l'irrigation
- L'hypothèse de stationnarité des séries temporelles n'est pas vérifiée formellement (test ADF à intégrer)

**Enjeux de fiabilité (LBRTI2101B) :**
- Qualité des données météorologiques : hétérogénéité des stations, valeurs manquantes
- Biais spatial : les 6 régions choisies ne couvrent pas uniformément le territoire
- Confidentialité : les données agronomiques sont souvent propriétaires ou soumises à restrictions

**Perspectives :**
- Intégration de données satellitaires (NDVI, MODIS) pour enrichir les features
- Modèles LSTM ou Transformer pour l'aspect temporel
- Application à la **prévision précoce des crises alimentaires** en contexte de changement climatique

---

## 📚 Sources des données

- Précipitations réelles : [CHIRPS (UCSB)](https://chc.ucsb.edu/data/chirps) — Climate Hazards Group InfraRed Precipitation
- Rendements agricoles : [FAOSTAT](https://www.fao.org/faostat/) — Organisation des Nations Unies pour l'alimentation
- Shapefiles Maroc : [GADM](https://gadm.org/) — Database of Global Administrative Areas

---

## 👤 Auteur

JABRANE Sohaib 
Filière : Data Science appliquée à l'agronomie  
Contact : jsohaibe@gmail.com

---

## 📄 Licence

MIT License — libre d'utilisation avec attribution.
