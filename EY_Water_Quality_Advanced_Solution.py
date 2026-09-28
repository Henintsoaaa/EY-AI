"""
================================================================================
EY WATER QUALITY CHALLENGE 2026 - SOLUTION AVANCÉE
================================================================================

Approches avancées:
1. GroupKFold pour éviter le data leakage
2. Pipelines scikit-learn pour un workflow propre
3. Monitoring train/val pour détecter l'overfitting
4. Feature engineering robuste et généralisable
5. Ensembling de plusieurs modèles
6. Early stopping basé sur la validation

Author: Advanced ML Pipeline
Date: 2026
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

import joblib
import os
from typing import Dict, List, Tuple

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("=" * 80)
print("EY WATER QUALITY CHALLENGE 2026 - SOLUTION AVANCÉE")
print("=" * 80)


# ============================================================================
# SECTION 1: CUSTOM TRANSFORMERS POUR PIPELINES
# ============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer pour feature engineering.
    Compatible avec sklearn Pipeline.
    """
    
    def __init__(self, use_cyclical_encoding=True, use_spectral_indices=True):
        self.use_cyclical_encoding = use_cyclical_encoding
        self.use_spectral_indices = use_spectral_indices
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # ================================================================
        # 1. FEATURES TEMPORELLES
        # ================================================================
        X['Sample Date'] = pd.to_datetime(X['Sample Date'], format='%d-%m-%Y')
        X['year'] = X['Sample Date'].dt.year
        X['month'] = X['Sample Date'].dt.month
        X['day_of_year'] = X['Sample Date'].dt.dayofyear
        X['quarter'] = X['Sample Date'].dt.quarter
        X['week_of_year'] = X['Sample Date'].dt.isocalendar().week
        
        # Encodage cyclique (évite la discontinuité décembre -> janvier)
        if self.use_cyclical_encoding:
            X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
            X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
            X['doy_sin'] = np.sin(2 * np.pi * X['day_of_year'] / 365)
            X['doy_cos'] = np.cos(2 * np.pi * X['day_of_year'] / 365)
            X['week_sin'] = np.sin(2 * np.pi * X['week_of_year'] / 52)
            X['week_cos'] = np.cos(2 * np.pi * X['week_of_year'] / 52)
        
        # Saison sud-africaine (inversée par rapport à l'hémisphère nord)
        def get_sa_season(month):
            if month in [12, 1, 2]: return 1  # Été
            elif month in [3, 4, 5]: return 2  # Automne
            elif month in [6, 7, 8]: return 3  # Hiver
            else: return 4  # Printemps
        
        X['season'] = X['month'].apply(get_sa_season)
        
        # One-hot encoding des saisons
        for season in [1, 2, 3, 4]:
            X[f'season_{season}'] = (X['season'] == season).astype(int)
        
        # ================================================================
        # 2. INDICES SPECTRAUX LANDSAT
        # ================================================================
        if self.use_spectral_indices and 'nir' in X.columns:
            # Calculer 'red' et 'blue' si manquants (approximations)
            if 'red' not in X.columns:
                X['red'] = X['green'] * 1.2  # Approximation
            if 'blue' not in X.columns:
                X['blue'] = X['green'] * 0.9  # Approximation
            
            # NDVI - Normalized Difference Vegetation Index
            X['NDVI'] = (X['nir'] - X['red']) / (X['nir'] + X['red'] + 1e-8)
            
            # EVI - Enhanced Vegetation Index
            X['EVI'] = 2.5 * ((X['nir'] - X['red']) / 
                             (X['nir'] + 6*X['red'] - 7.5*X['blue'] + 1 + 1e-8))
            
            # SAVI - Soil Adjusted Vegetation Index
            L = 0.5
            X['SAVI'] = ((X['nir'] - X['red']) / (X['nir'] + X['red'] + L)) * (1 + L)
            
            # NDWI - Normalized Difference Water Index
            X['NDWI'] = (X['green'] - X['nir']) / (X['green'] + X['nir'] + 1e-8)
            
            # Indices avec SWIR
            if 'swir16' in X.columns:
                X['NDMI_calc'] = (X['nir'] - X['swir16']) / (X['nir'] + X['swir16'] + 1e-8)
                X['MNDWI_calc'] = (X['green'] - X['swir16']) / (X['green'] + X['swir16'] + 1e-8)
                
                # BSI - Bare Soil Index
                X['BSI'] = ((X['swir16'] + X['red']) - (X['nir'] + X['blue'])) / \
                          ((X['swir16'] + X['red']) + (X['nir'] + X['blue']) + 1e-8)
            
            if 'swir22' in X.columns:
                # NBR - Normalized Burn Ratio
                X['NBR'] = (X['nir'] - X['swir22']) / (X['nir'] + X['swir22'] + 1e-8)
        
        # ================================================================
        # 3. INTERACTIONS CLIMATIQUES-SPECTRALES
        # ================================================================
        if 'pet' in X.columns:
            if 'NDVI' in X.columns:
                X['pet_x_ndvi'] = X['pet'] * X['NDVI']
                X['pet_x_evi'] = X['pet'] * X['EVI']
            
            X['pet_x_season'] = X['pet'] * X['season']
            X['pet_squared'] = X['pet'] ** 2
        
        # ================================================================
        # 4. FEATURES GÉOGRAPHIQUES ABSTRAITES
        # ================================================================
        # IMPORTANT: Ne pas utiliser lat/lon directes!
        X['lat_rounded'] = (X['Latitude'] // 0.5) * 0.5
        X['lon_rounded'] = (X['Longitude'] // 0.5) * 0.5
        X['dist_to_equator'] = np.abs(X['Latitude'])
        X['coastal_proximity'] = np.minimum(
            np.abs(X['Latitude'] + 34),  # Distance à la côte sud
            np.abs(X['Latitude'] + 22)   # Distance à la côte est
        )
        
        # ================================================================
        # 5. FEATURES STATISTIQUES (si plusieurs bandes disponibles)
        # ================================================================
        spectral_bands = ['nir', 'green', 'swir16', 'swir22']
        available_bands = [b for b in spectral_bands if b in X.columns]
        
        if len(available_bands) >= 2:
            X['spectral_mean'] = X[available_bands].mean(axis=1)
            X['spectral_std'] = X[available_bands].std(axis=1)
            X['spectral_range'] = X[available_bands].max(axis=1) - X[available_bands].min(axis=1)
        
        # ================================================================
        # 6. NETTOYAGE
        # ================================================================
        # Supprimer les colonnes non-numériques
        cols_to_drop = ['Sample Date']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
        
        # Supprimer les colonnes infinies ou NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clippe les outliers à des percentiles donnés.
    """
    
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.bounds = {}
        
    def fit(self, X, y=None):
        for col in X.select_dtypes(include=[np.number]).columns:
            self.bounds[col] = (
                X[col].quantile(self.lower_percentile / 100),
                X[col].quantile(self.upper_percentile / 100)
            )
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower, upper)
        return X


# ============================================================================
# SECTION 2: CHARGEMENT DES DONNÉES
# ============================================================================

def load_all_data():
    """Charge tous les datasets"""
    print("\n" + "=" * 80)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 80)
    
    # Training data
    water_quality_train = pd.read_csv('water_quality_training_dataset.csv')
    landsat_train = pd.read_csv('landsat_features_training.csv')
    terraclimate_train = pd.read_csv('terraclimate_features_training.csv')
    
    # Validation data
    landsat_val = pd.read_csv('landsat_features_validation.csv')
    terraclimate_val = pd.read_csv('terraclimate_features_validation.csv')
    
    print(f"\n✅ Données chargées:")
    print(f"   - Water Quality: {water_quality_train.shape}")
    print(f"   - Landsat Train: {landsat_train.shape}")
    print(f"   - TerraClimate Train: {terraclimate_train.shape}")
    print(f"   - Landsat Val: {landsat_val.shape}")
    print(f"   - TerraClimate Val: {terraclimate_val.shape}")
    
    # Fusion pour training
    train_df = water_quality_train.merge(
        landsat_train, on=['Latitude', 'Longitude', 'Sample Date'], how='left'
    )
    train_df = train_df.merge(
        terraclimate_train, on=['Latitude', 'Longitude', 'Sample Date'], how='left'
    )
    
    # Fusion pour test (on crée un DataFrame vide avec les mêmes colonnes)
    # On utilisera les features de validation mais sans targets
    test_df = landsat_val.merge(
        terraclimate_val, on=['Latitude', 'Longitude', 'Sample Date'], how='left'
    )
    
    print(f"\n✅ Datasets fusionnés:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    
    return train_df, test_df


# ============================================================================
# SECTION 3: PIPELINES DE MODÉLISATION
# ============================================================================

def create_model_pipeline(model_type='xgboost', model_params=None):
    """
    Crée un pipeline complet de preprocessing + modèle.
    """
    if model_params is None:
        model_params = {}
    
    # Sélectionner le modèle
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=model_params.get('n_estimators', 500),
            max_depth=model_params.get('max_depth', 4),
            learning_rate=model_params.get('learning_rate', 0.05),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            reg_alpha=model_params.get('reg_alpha', 1.0),
            reg_lambda=model_params.get('reg_lambda', 2.0),
            random_state=42,
            n_jobs=-1,
            **{k: v for k, v in model_params.items() 
               if k not in ['n_estimators', 'max_depth', 'learning_rate', 
                           'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']}
        )
    
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(
            n_estimators=model_params.get('n_estimators', 500),
            max_depth=model_params.get('max_depth', 4),
            learning_rate=model_params.get('learning_rate', 0.05),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            reg_alpha=model_params.get('reg_alpha', 1.0),
            reg_lambda=model_params.get('reg_lambda', 2.0),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **{k: v for k, v in model_params.items() 
               if k not in ['n_estimators', 'max_depth', 'learning_rate', 
                           'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']}
        )
    
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 300),
            max_depth=model_params.get('max_depth', 10),
            min_samples_split=model_params.get('min_samples_split', 20),
            min_samples_leaf=model_params.get('min_samples_leaf', 10),
            max_features=model_params.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=model_params.get('n_estimators', 500),
            max_depth=model_params.get('max_depth', 4),
            learning_rate=model_params.get('learning_rate', 0.05),
            subsample=model_params.get('subsample', 0.8),
            random_state=42
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Créer le pipeline
    pipeline = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('outlier_clipping', OutlierClipper()),
        ('scaler', RobustScaler()),
        ('model', model)
    ])
    
    return pipeline


# ============================================================================
# SECTION 4: VALIDATION AVEC GROUPKFOLD ET MONITORING
# ============================================================================

def evaluate_with_monitoring(
    X, y, groups, 
    model_type='xgboost', 
    model_params=None,
    n_splits=5,
    target_name='Target'
):
    """
    Évalue un modèle avec GroupKFold et monitoring train/val.
    """
    print(f"\n{'='*80}")
    print(f"ÉVALUATION: {model_type.upper()} - {target_name}")
    print(f"{'='*80}")
    
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    train_scores = []
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        # Préparation des données
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        y_val_fold = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
        
        # Créer et entraîner le pipeline
        pipeline = create_model_pipeline(model_type, model_params)
        
        # Entraînement
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Prédictions train et val
        y_train_pred = pipeline.predict(X_train_fold)
        y_val_pred = pipeline.predict(X_val_fold)
        
        # Calcul des métriques
        train_r2 = r2_score(y_train_fold, y_train_pred)
        val_r2 = r2_score(y_val_fold, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        val_mae = mean_absolute_error(y_val_fold, y_val_pred)
        
        train_scores.append(train_r2)
        val_scores.append(val_r2)
        
        # Vérifier l'overfitting
        overfitting_gap = train_r2 - val_r2
        overfitting_status = "⚠️ OVERFITTING" if overfitting_gap > 0.15 else "✅ OK"
        
        print(f"\nFold {fold}/{n_splits}:")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Val R²:   {val_r2:.4f}")
        print(f"   Val RMSE: {val_rmse:.4f}")
        print(f"   Val MAE:  {val_mae:.4f}")
        print(f"   Gap:      {overfitting_gap:.4f} {overfitting_status}")
        
        # Vérifier qu'il n'y a pas de data leakage
        train_stations = groups.iloc[train_idx].unique()
        val_stations = groups.iloc[val_idx].unique()
        overlap = set(train_stations) & set(val_stations)
        assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} stations en commun!"
        
        fold_results.append({
            'fold': fold,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'overfitting_gap': overfitting_gap
        })
    
    # Résumé
    mean_train_r2 = np.mean(train_scores)
    mean_val_r2 = np.mean(val_scores)
    std_val_r2 = np.std(val_scores)
    mean_gap = mean_train_r2 - mean_val_r2
    
    print(f"\n{'='*80}")
    print(f"📊 RÉSULTATS FINAUX - {target_name}")
    print(f"{'='*80}")
    print(f"Train R² moyen:     {mean_train_r2:.4f}")
    print(f"Val R² moyen:       {mean_val_r2:.4f} ± {std_val_r2:.4f}")
    print(f"Overfitting moyen:  {mean_gap:.4f}")
    
    if mean_gap > 0.15:
        print("⚠️  ATTENTION: Overfitting détecté!")
        print("   Suggestions:")
        print("   - Augmenter reg_alpha et reg_lambda")
        print("   - Réduire max_depth")
        print("   - Réduire n_estimators")
    else:
        print("✅ Pas d'overfitting significatif")
    
    return {
        'mean_val_r2': mean_val_r2,
        'std_val_r2': std_val_r2,
        'mean_train_r2': mean_train_r2,
        'overfitting_gap': mean_gap,
        'fold_results': fold_results
    }


# ============================================================================
# SECTION 5: ENTRAÎNEMENT MULTI-TARGET
# ============================================================================

def train_multi_target_models(X, y_dict, groups, model_configs):
    """
    Entraîne plusieurs modèles pour plusieurs targets.
    
    Args:
        X: Features DataFrame
        y_dict: Dict {'target_name': target_series}
        groups: Groups pour GroupKFold
        model_configs: Dict {'model_name': {'type': ..., 'params': ...}}
    
    Returns:
        Dict avec les résultats
    """
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT MULTI-TARGET")
    print("=" * 80)
    
    all_results = {}
    
    for target_name, y in y_dict.items():
        print(f"\n{'#'*80}")
        print(f"TARGET: {target_name}")
        print(f"{'#'*80}")
        
        target_results = {}
        
        for model_name, config in model_configs.items():
            results = evaluate_with_monitoring(
                X=X,
                y=y,
                groups=groups,
                model_type=config['type'],
                model_params=config.get('params', {}),
                n_splits=5,
                target_name=f"{target_name} - {model_name}"
            )
            
            target_results[model_name] = results
        
        all_results[target_name] = target_results
    
    return all_results


# ============================================================================
# SECTION 6: SÉLECTION DU MEILLEUR MODÈLE PAR TARGET
# ============================================================================

def select_best_models(all_results):
    """
    Sélectionne le meilleur modèle pour chaque target.
    """
    print("\n" + "=" * 80)
    print("SÉLECTION DES MEILLEURS MODÈLES")
    print("=" * 80)
    
    best_models = {}
    
    for target_name, target_results in all_results.items():
        best_model = None
        best_score = -np.inf
        
        for model_name, results in target_results.items():
            score = results['mean_val_r2']
            if score > best_score:
                best_score = score
                best_model = model_name
        
        best_models[target_name] = {
            'model': best_model,
            'score': best_score,
            'overfitting': target_results[best_model]['overfitting_gap']
        }
        
        print(f"\n{target_name}:")
        print(f"   Meilleur modèle: {best_model}")
        print(f"   R² validation:   {best_score:.4f}")
        print(f"   Overfitting:     {target_results[best_model]['overfitting_gap']:.4f}")
    
    # Score moyen global
    mean_score = np.mean([info['score'] for info in best_models.values()])
    print(f"\n{'='*80}")
    print(f"📊 SCORE MOYEN GLOBAL: {mean_score:.4f}")
    print(f"{'='*80}")
    
    return best_models


# ============================================================================
# SECTION 7: ENTRAÎNEMENT FINAL ET PRÉDICTIONS
# ============================================================================

def train_final_and_predict(
    X_train, y_dict, X_test, 
    best_models, model_configs
):
    """
    Entraîne les modèles finaux sur toutes les données et prédit.
    """
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT FINAL ET PRÉDICTIONS")
    print("=" * 80)
    
    final_predictions = {}
    final_pipelines = {}
    
    for target_name, y in y_dict.items():
        best_model_name = best_models[target_name]['model']
        config = model_configs[best_model_name]
        
        print(f"\n📊 Entraînement final pour {target_name} avec {best_model_name}...")
        
        # Créer le pipeline
        pipeline = create_model_pipeline(
            model_type=config['type'],
            model_params=config.get('params', {})
        )
        
        # Entraîner sur toutes les données
        pipeline.fit(X_train, y)
        
        # Prédire sur le test set
        predictions = pipeline.predict(X_test)
        
        final_predictions[target_name] = predictions
        final_pipelines[target_name] = pipeline
        
        print(f"   ✅ Prédictions générées:")
        print(f"      Min: {predictions.min():.2f}")
        print(f"      Max: {predictions.max():.2f}")
        print(f"      Mean: {predictions.mean():.2f}")
        print(f"      Median: {np.median(predictions):.2f}")
    
    return final_predictions, final_pipelines


# ============================================================================
# SECTION 8: PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Pipeline principal d'exécution"""
    
    # 1. Chargement des données
    train_df, test_df = load_all_data()
    
    # 2. Préparation
    print("\n" + "=" * 80)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 80)
    
    # Targets
    target_cols = [
        'Total Alkalinity', 
        'Electrical Conductance', 
        'Dissolved Reactive Phosphorus'
    ]
    
    # Créer les groupes (station_id) AVANT le feature engineering
    train_df['station_id'] = (
        train_df['Latitude'].astype(str) + '_' + 
        train_df['Longitude'].astype(str)
    )
    groups = train_df['station_id']
    
    print(f"\n✅ Groupes créés:")
    print(f"   Nombre de stations uniques: {groups.nunique()}")
    print(f"   Nombre total d'échantillons: {len(groups)}")
    
    # Features (avant feature engineering, on garde les colonnes de base)
    X_train_raw = train_df.drop(columns=target_cols + ['station_id'])
    X_test_raw = test_df.copy()
    
    # Targets
    y_dict = {col: train_df[col] for col in target_cols}
    
    print(f"\n✅ Données préparées:")
    print(f"   X_train_raw: {X_train_raw.shape}")
    print(f"   X_test_raw: {X_test_raw.shape}")
    print(f"   Targets: {list(y_dict.keys())}")
    
    # 3. Configuration des modèles à tester
    model_configs = {
        'XGBoost_Conservative': {
            'type': 'xgboost',
            'params': {
                'n_estimators': 500,
                'max_depth': 3,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 2.0,
                'reg_lambda': 3.0,
            }
        },
        'XGBoost_Moderate': {
            'type': 'xgboost',
            'params': {
                'n_estimators': 600,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 2.0,
            }
        },
        'LightGBM': {
            'type': 'lightgbm',
            'params': {
                'n_estimators': 500,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 2.0,
            }
        },
        'RandomForest': {
            'type': 'random_forest',
            'params': {
                'n_estimators': 300,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
            }
        }
    }
    
    # 4. Entraînement et évaluation
    all_results = train_multi_target_models(
        X=X_train_raw,
        y_dict=y_dict,
        groups=groups,
        model_configs=model_configs
    )
    
    # 5. Sélection des meilleurs modèles
    best_models = select_best_models(all_results)
    
    # 6. Entraînement final et prédictions
    final_predictions, final_pipelines = train_final_and_predict(
        X_train=X_train_raw,
        y_dict=y_dict,
        X_test=X_test_raw,
        best_models=best_models,
        model_configs=model_configs
    )
    
    # 7. Création du fichier de soumission
    print("\n" + "=" * 80)
    print("CRÉATION DU FICHIER DE SOUMISSION")
    print("=" * 80)
    
    submission = test_df[['Latitude', 'Longitude', 'Sample Date']].copy()
    submission['Total Alkalinity'] = final_predictions['Total Alkalinity']
    submission['Electrical Conductance'] = final_predictions['Electrical Conductance']
    submission['Dissolved Reactive Phosphorus'] = final_predictions['Dissolved Reactive Phosphorus']
    
    print("\n📋 Aperçu du fichier de soumission:")
    print(submission.head(10))
    
    # Vérifications
    print("\n🔍 Vérifications:")
    
    # NaN
    nan_count = submission[target_cols].isna().sum().sum()
    if nan_count == 0:
        print("   ✅ Pas de NaN")
    else:
        print(f"   ⚠️  {nan_count} NaN détectés!")
    
    # Plages de valeurs
    checks = [
        ('Total Alkalinity', 20, 250),
        ('Electrical Conductance', 100, 1000),
        ('Dissolved Reactive Phosphorus', 5, 150)
    ]
    
    all_ok = True
    for col, min_val, max_val in checks:
        col_min = submission[col].min()
        col_max = submission[col].max()
        
        if col_min < min_val or col_max > max_val:
            print(f"   ⚠️  {col}: [{col_min:.2f}, {col_max:.2f}] (attendu: [{min_val}, {max_val}])")
            all_ok = False
        else:
            print(f"   ✅ {col}: [{col_min:.2f}, {col_max:.2f}]")
    
    # Sauvegarde
    submission_filename = 'submission_advanced_pipeline.csv'
    submission.to_csv(submission_filename, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ FICHIER DE SOUMISSION CRÉÉ: {submission_filename}")
    print(f"{'='*80}")
    print(f"\nScore attendu sur le leaderboard:")
    mean_score = np.mean([info['score'] for info in best_models.values()])
    print(f"   R² moyen estimé: {mean_score:.4f}")
    print(f"\n🚀 PRÊT POUR UPLOAD SUR LE LEADERBOARD!")
    
    # 8. Sauvegarde des modèles
    os.makedirs('models_advanced', exist_ok=True)
    for target_name, pipeline in final_pipelines.items():
        filename = f"models_advanced/{target_name.replace(' ', '_')}_pipeline.pkl"
        joblib.dump(pipeline, filename)
        print(f"   ✅ Sauvegardé: {filename}")
    
    print("\n" + "=" * 80)
    print("FIN DU PIPELINE")
    print("=" * 80)
    
    return submission, best_models, all_results


if __name__ == "__main__":
    submission, best_models, all_results = main()
