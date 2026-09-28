# EY Landsat TerraClimate Water Quality

Projet de prévision de paramètres de qualité de l'eau à partir de mesures environnementales, de données Landsat et de variables TerraClimate.

## Objectif

Prédire plusieurs indicateurs de qualité de l'eau :

- Total Alkalinity
- Electrical Conductance
- Dissolved Reactive Phosphorus

## Données

- Mesures de qualité de l'eau
- Variables satellitaires Landsat
- Variables climatiques TerraClimate
- Coordonnées géographiques et dates d'échantillonnage

## Méthodes

- Fusion de sources multi-capteurs et multi-temporelles
- Variables temporelles et saisonnières
- Indices spectraux : NDVI, EVI, SAVI, NDWI et NBR
- Feature engineering géospatial
- Gestion des valeurs aberrantes
- Validation GroupKFold
- Comparaison de Ridge, Random Forest, Gradient Boosting, XGBoost et LightGBM
- Ensembles de modèles et sauvegarde des pipelines

## Structure

- `code/` : données et notebooks du challenge
- `Jupyter Notebook Package/` : extraction et démonstration Landsat/TerraClimate
- `models/` et `models_advanced/` : modèles et pipelines sauvegardés
- `EY_Water_Quality_Advanced_Solution.py` : pipeline avancé
- `submission_advanced_pipeline.csv` : prédictions produites

## Reproductibilité

Les chemins de données utilisés par certains notebooks correspondent à l'organisation du dossier `code/`. Les dépendances principales sont Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, GeoPandas, Rasterio et Xarray.
