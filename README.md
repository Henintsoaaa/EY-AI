# EY Landsat TerraClimate Water Quality

Projet de prévision de paramètres de qualité de l'eau à partir de mesures environnementales, de données Landsat et de variables TerraClimate.

## Objectif

Prédire plusieurs indicateurs de qualité de l'eau :

## Données

## Méthodes

## Structure

## Structure

- `data/raw/` : données qualité de l'eau, Landsat et TerraClimate
- `notebooks/data_extraction/` : extraction et démonstration des données environnementales
- `notebooks/water_quality/` : variantes des notebooks de modélisation
- `notebooks/experiments/` : expérience séparée de classification CIFAR
- `src/ey_water_quality_pipeline.py` : pipeline avancé reproductible
- `models/baseline/` : modèles de référence
- `models/ensemble/` : modèles ensemblistes
- `models/advanced/` : pipelines avancés par cible
- `submissions/` : prédictions générées et templates
- `docs/` : guide du challenge et métadonnées
- `submission_advanced_pipeline.csv` : prédictions produites

## Reproductibilité

Les notebooks de modélisation doivent être exécutés depuis leur emplacement dans `notebooks/water_quality/`. Le pipeline Python principal peut être lancé depuis n'importe quel répertoire :

```bash
python3 src/ey_water_quality_pipeline.py
```

Les dépendances principales sont Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, Joblib, GeoPandas, Rasterio et Xarray.
Les chemins de données utilisés par certains notebooks correspondent à l'organisation du dossier `code/`. Les dépendances principales sont Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, GeoPandas, Rasterio et Xarray.
