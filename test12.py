#!/usr/bin/env python
# coding: utf-8


# 13. MACHINE LEARNING - MODÈLES PREDICTIFS


print("\n" + "="*80)
print("13. MACHINE LEARNING - MODÈLES PRÉDICTIFS DU NOMBRE IDÉAL D'ENFANTS")
print("="*80)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


# 13.1 PRÉPARATION DES DONNÉES - VARIABLES DU MODÈLE FINAL (10 variables)

print("\n" + "-"*60)
print("13.1 PRÉPARATION DES DONNÉES")
print("-"*60)

variables_ml = {
    'catégorielles': ['edu_cat', 'richesse_cat', 'region_cat', 
                      'besoin_pf_ns', 'edu_mari_cat', 'religion_cat',
                      'proprietaire_terre_cat', 'regarde_tv_cat', 'regarde_journal_cat'],
    'continues': ['nb_vivants_centre']
}

print("Variables du modèle final (10 variables):")
print(f"  Catégorielles ({len(variables_ml['catégorielles'])}): {variables_ml['catégorielles']}")
print(f"  Continues ({len(variables_ml['continues'])}): {variables_ml['continues']}")

df_ml = df_model[variables_ml['catégorielles'] + variables_ml['continues'] + ['Y']].copy()

print(f"\nDimensions: {df_ml.shape[0]} observations, {df_ml.shape[1]-1} features")

X = df_ml.drop('Y', axis=1)
y = df_ml['Y']

cat_features = variables_ml['catégorielles']
num_features = variables_ml['continues']

print(f"\nFeatures catégorielles: {cat_features}")
print(f"Features numériques: {num_features}")


# 13.2 PIPELINE DE PRÉTRAITEMENT


print("\n" + "-"*60)
print("13.2 PIPELINE DE PRÉTRAITEMENT")
print("-"*60)

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

print("Prétraitement appliqué:")
print("  - Variables catégorielles: One-Hot Encoding (drop='first')")
print("  - Variables continues: StandardScaler (centrage-réduction)")
print("  - Gestion des valeurs inconnues: handle_unknown='ignore'")


# 13.3 DIVISION TRAIN / VALIDATION / TEST


print("\n" + "-"*60)
print("13.3 DIVISION ENSEMBLES TRAIN / VALIDATION / TEST")
print("-"*60)

# Étape 1 : Séparation Test (20%) — JAMAIS touché avant l'évaluation finale
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=None
)

# Étape 2 : Séparation Train (60% total) / Validation (20% total)
# 0.25 de 80% = 20% du total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=None
)

print(f"Train:       {X_train.shape[0]} observations ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation:  {X_val.shape[0]} observations ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:        {X_test.shape[0]} observations ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nDistribution de Y (train):       moy={y_train.mean():.2f}, std={y_train.std():.2f}")
print(f"Distribution de Y (validation):  moy={y_val.mean():.2f}, std={y_val.std():.2f}")
print(f"Distribution de Y (test):        moy={y_test.mean():.2f}, std={y_test.std():.2f}")


#  DÉFINITION DES 6 MODÈLES


print("\n" + "-"*60)
print("13.4 DÉFINITION DES 6 MODÈLES DE MACHINE LEARNING")
print("-"*60)

modeles = {
    'Ridge': {
        'pipeline': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=1.0, random_state=42))
        ]),
        'params': {'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'Lasso': {
        'pipeline': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(alpha=0.1, random_state=42, max_iter=5000))
        ]),
        'params': {'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    },
    'ElasticNet': {
        'pipeline': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000))
        ]),
        'params': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    },
    'DecisionTree': {
        'pipeline': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ]),
        'params': {
            'regressor__max_depth': [3, 5, 7, 10, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
    },
    'RandomForest': {
        'pipeline': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'params': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [5, 10, 15, None],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2]
        }
    },
    'GradientBoosting': {
        'pipeline': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ]),
        'params': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__subsample': [0.8, 1.0]
        }
    }
}

for nom, config in modeles.items():
    print(f"\n{nom}:")
    print(f"  Pipeline: {list(config['pipeline'].named_steps.keys())}")
    print(f"  Hyperparamètres à optimiser: {list(config['params'].keys())}")


# 13.5 FONCTION D'ÉVALUATION (TRAIN / VALIDATION / TEST)

print("\n" + "-"*60)
print("13.5 FONCTION D'ÉVALUATION COMPLÈTE (3 SETS)")
print("-"*60)

def evaluer_modele_ml(nom, pipeline, X_train, y_train, X_val, y_val, X_test, y_test, deja_fit=False):
    """
    Évalue un modèle de ML sur Train, Validation et Test.
    """
    if not deja_fit:
        pipeline.fit(X_train, y_train)
    
    # Prédictions sur les 3 ensembles
    y_pred_train = pipeline.predict(X_train)
    y_pred_val   = pipeline.predict(X_val)
    y_pred_test  = pipeline.predict(X_test)
    
    # Métriques RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val   = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Métriques Test
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    r2_test   = r2_score(y_test, y_pred_test)
    
    # MAPE (sur test, valeurs > 0 uniquement)
    y_test_nz = y_test[y_test > 0]
    y_pred_nz = y_pred_test[y_test > 0]
    mape = np.mean(np.abs((y_test_nz - y_pred_nz) / y_test_nz)) * 100 if len(y_test_nz) > 0 else np.nan
    
    # Accuracy à ±0.5 et ±1 près (sur test)
    acc_half = np.mean(np.abs(y_test - y_pred_test) < 0.5) * 100
    acc_one  = np.mean(np.abs(y_test - y_pred_test) < 1.0) * 100
    
    # Sur-apprentissage = écart entre Train et Validation (JAMAIS Test)
    ecart_rmse = rmse_train - rmse_val
    
    print(f"\n{'='*60}")
    print(f" {nom}")
    print(f"{'='*60}")
    print(f"RMSE (train)       : {rmse_train:.4f}")
    print(f"RMSE (validation)  : {rmse_val:.4f}")
    print(f"RMSE (test)        : {rmse_test:.4f}")
    print(f"MAE (test)         : {mae_test:.4f}")
    print(f"R² (test)          : {r2_test:.4f}")
    print(f"MAPE (test)        : {mape:.2f}%")
    print(f"Accuracy ±0.5      : {acc_half:.1f}%")
    print(f"Accuracy ±1.0      : {acc_one:.1f}%")
    print(f"Écart train/val    : {ecart_rmse:.4f}")
    
    if abs(ecart_rmse) > 0.3:
        print("  Sur-apprentissage détecté (train vs validation)!")
    else:
        print(" Pas de sur-apprentissage significatif")
    
    return {
        'nom': nom,
        'pipeline': pipeline,
        'rmse_train': rmse_train,
        'rmse_val': rmse_val,
        'rmse_test': rmse_test,
        'mae': mae_test,
        'r2': r2_test,
        'mape': mape,
        'acc_half': acc_half,
        'acc_one': acc_one,
        'ecart': ecart_rmse,
        'y_pred_test': y_pred_test,
        'y_pred_val': y_pred_val,
        'y_pred_train': y_pred_train
    }



print("\n" + "-"*60)
print("13.6 ENTRAÎNEMENT (GRIDSEARCH SUR TRAIN) ET ÉVALUATION (TRAIN/VAL/TEST)")
print("-"*60)

resultats_ml = {}

for nom, config in modeles.items():
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT: {nom}")
    print(f"{'='*70}")
    
    # GridSearchCV : optimisation sur X_train UNIQUEMENT (CV interne = folds du train)
    grid = GridSearchCV(
        config['pipeline'],
        config['params'],
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres: {grid.best_params_}")
    print(f"Meilleur RMSE CV (train folds): {-grid.best_score_:.4f}")
    
    # Évaluation sur les 3 sets avec le meilleur modèle (déjà fitté sur X_train)
    meilleur_pipeline = grid.best_estimator_
    resultats_ml[nom] = evaluer_modele_ml(
        nom, meilleur_pipeline, 
        X_train, y_train, X_val, y_val, X_test, y_test, 
        deja_fit=True
    )
    resultats_ml[nom]['best_params'] = grid.best_params_
    resultats_ml[nom]['cv_rmse'] = -grid.best_score_



print("\n" + "="*80)
print("13.7 CLASSEMENT FINAL DES 6 MODÈLES (BASÉ SUR VALIDATION)")
print("="*80)

comparaison = pd.DataFrame({
    'Modèle': list(resultats_ml.keys()),
    'RMSE CV': [resultats_ml[m]['cv_rmse'] for m in resultats_ml],
    'RMSE Train': [resultats_ml[m]['rmse_train'] for m in resultats_ml],
    'RMSE Val': [resultats_ml[m]['rmse_val'] for m in resultats_ml],
    'RMSE Test': [resultats_ml[m]['rmse_test'] for m in resultats_ml],
    'MAE Test': [resultats_ml[m]['mae'] for m in resultats_ml],
    'R² Test': [resultats_ml[m]['r2'] for m in resultats_ml],
    'MAPE (%)': [resultats_ml[m]['mape'] for m in resultats_ml],
    'Acc ±0.5 (%)': [resultats_ml[m]['acc_half'] for m in resultats_ml],
    'Acc ±1.0 (%)': [resultats_ml[m]['acc_one'] for m in resultats_ml],
    'Sur-app.': ['Oui' if abs(resultats_ml[m]['ecart']) > 0.3 else 'Non' for m in resultats_ml]
})

# Classement par RMSE Validation (sélection du modèle)
comparaison = comparaison.sort_values('RMSE Val')
print("\n" + comparaison.to_string(index=False))

# Identification du meilleur modèle (basé sur la validation)
meilleur_nom = comparaison.iloc[0]['Modèle']
print(f"\n MEILLEUR MODÈLE (sur validation): {meilleur_nom}")
print(f"   RMSE Validation = {resultats_ml[meilleur_nom]['rmse_val']:.4f}")
print(f"   RMSE Test       = {resultats_ml[meilleur_nom]['rmse_test']:.4f}")
print(f"   R² Test         = {resultats_ml[meilleur_nom]['r2']:.4f}")
print(f"   Accuracy ±1.0   = {resultats_ml[meilleur_nom]['acc_one']:.1f}%")


# 13.8 VISUALISATIONS COMPARATIVES


print("\n" + "-"*60)
print("13.8 VISUALISATIONS COMPARATIVES")
print("-"*60)

# Figure 1: Comparaison des métriques
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

metrics = ['RMSE Val', 'RMSE Test', 'MAE Test', 'R² Test', 'Acc ±0.5 (%)', 'Acc ±1.0 (%)']
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(comparaison)))

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    bars = ax.barh(comparaison['Modèle'], comparaison[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f'Comparaison: {metric}')
    
    for bar, val in zip(bars, comparaison[metric]):
        if not np.isnan(val):
            ax.text(val + 0.01 * max(comparaison[metric]), bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
    
    # Meilleur modèle en vert (min pour erreurs, max pour R²/acc)
    if 'RMSE' in metric or 'MAE' in metric or 'MAPE' in metric:
        best_idx = comparaison[metric].idxmin()
    else:
        best_idx = comparaison[metric].idxmax()
    bars[best_idx].set_color('#2ecc71')

plt.tight_layout()
plt.savefig('ml_comparaison_metriques.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: ml_comparaison_metriques.png")

# Figure 2: Prédictions vs Réel sur le TEST SET (meilleur modèle)
y_pred_best = resultats_ml[meilleur_nom]['y_pred_test']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
axes[0].scatter(y_test, y_pred_best, alpha=0.4, s=20, edgecolors='none', color='#3498db')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Parfaite')
axes[0].set_xlabel('Valeur réelle (Y)', fontsize=12)
axes[0].set_ylabel('Valeur prédite (Ŷ)', fontsize=12)
axes[0].set_title(f'{meilleur_nom}: Prédictions vs Réel (TEST SET)\nR² = {resultats_ml[meilleur_nom]["r2"]:.4f}', 
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution des résidus (sur test)
residus = y_test - y_pred_best
axes[1].hist(residus, bins=50, edgecolor='black', color='#e74c3c', alpha=0.7)
axes[1].axvline(x=0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel('Résidus (Y - Ŷ)', fontsize=12)
axes[1].set_ylabel('Fréquence', fontsize=12)
axes[1].set_title(f'Distribution des résidus (TEST SET)\nMoy={residus.mean():.3f}, Std={residus.std():.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_predictions_residus.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: ml_predictions_residus.png")


#  COURBES D'APPRENTISSAGE

print("\n" + "-"*60)
print("13.9 COURBES D'APPRENTISSAGE (TRAIN vs VALIDATION CV)")
print("-"*60)

# Courbes d'apprentissage pour les 3 meilleurs modèles (basés sur RMSE Val)
top3 = comparaison.head(3)['Modèle'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, nom in enumerate(top3):
    print(f"\nCalcul courbe d'apprentissage: {nom}...")
    
    pipeline = resultats_ml[nom]['pipeline']
    
    # CORRECTION : X_train uniquement ! La CV interne gère la validation.
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_root_mean_squared_error'
    )
    
    train_mean = -np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = -np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)
    
    ax = axes[idx]
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Train RMSE')
    ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation RMSE')
    ax.set_xlabel('N observations (train)', fontsize=10)
    ax.set_ylabel('RMSE', fontsize=10)
    ax.set_title(f'{nom}\nRMSE Val={resultats_ml[nom]["rmse_val"]:.3f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée: ml_learning_curves.png")

#IMPORTANCE DES VARIABLES (SHAP)


print("\n" + "-"*60)
print("13.10 ANALYSE SHAP (SHapley Additive exPlanations)")
print("-"*60)

try:
    import shap
    print("Package SHAP installé.")
    
    # Préparation : fit du préprocesseur sur X_train uniquement
    pipeline_meilleur = resultats_ml[meilleur_nom]['pipeline']
    
    # Clone et réentraînement sur X_train pour être sûr
    pipeline_shap = clone(pipeline_meilleur)
    pipeline_shap.fit(X_train, y_train)
    
    preprocessor_fitted = pipeline_shap.named_steps['preprocessor']
    X_train_processed = preprocessor_fitted.transform(X_train)
    X_test_processed  = preprocessor_fitted.transform(X_test)
    
    feature_names = (num_features + 
                    list(preprocessor_fitted.named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(cat_features)))
    
    print(f"Nombre de features après prétraitement: {len(feature_names)}")
    print(f"Features: {feature_names[:10]}...")
    
    regressor = pipeline_shap.named_steps['regressor']
    
    if hasattr(regressor, 'feature_importances_'):
        print(f"\nCalcul SHAP pour {meilleur_nom} (modèle basé sur arbres)...")
        
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_test_processed)
        
        # Summary plot global
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, 
                         show=False, max_display=20)
        plt.title(f'SHAP Summary Plot - {meilleur_nom} (Test Set)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('ml_shap_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure sauvegardée: ml_shap_summary.png")
        
        # Bar plot des importances moyennes
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names,
                         plot_type="bar", show=False, max_display=15)
        plt.title(f'Importance moyenne SHAP - {meilleur_nom} (Test Set)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('ml_shap_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure sauvegardée: ml_shap_importance.png")
        
        # Dependence plots pour les 3 features les plus importantes
        shap_importance = np.abs(shap_values).mean(0)
        top_features_idx = np.argsort(shap_importance)[-3:][::-1]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, feat_idx in enumerate(top_features_idx):
            shap.dependence_plot(feat_idx, shap_values, X_test_processed, 
                               feature_names=feature_names, ax=axes[idx], show=False)
            axes[idx].set_title(f'Dependence Plot: {feature_names[feat_idx]}', fontsize=10)
        plt.tight_layout()
        plt.savefig('ml_shap_dependence.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure sauvegardée: ml_shap_dependence.png")
        
    else:
        print(f"\nCalcul SHAP pour {meilleur_nom} (modèle linéaire)...")
        
        X_sample = shap.sample(X_test_processed, 100)
        explainer = shap.KernelExplainer(regressor.predict, X_sample)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         show=False, max_display=15)
        plt.title(f'SHAP Summary Plot - {meilleur_nom} (Test Set)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('ml_shap_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure sauvegardée: ml_shap_summary.png")

except ImportError:
    print(" Package SHAP non installé. Installation: pip install shap")
    print("Analyse SHAP ignorée.")
    
    print("\nImportance des features (méthode native):")
    regressor = resultats_ml[meilleur_nom]['pipeline'].named_steps['regressor']
    
    if hasattr(regressor, 'feature_importances_'):
        preprocessor_fitted = resultats_ml[meilleur_nom]['pipeline'].named_steps['preprocessor']
        feature_names = (num_features + 
                        list(preprocessor_fitted.named_transformers_['cat']
                            .named_steps['onehot']
                            .get_feature_names_out(cat_features)))
        
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': regressor.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importances.head(15).to_string(index=False))
        
        plt.figure(figsize=(12, 8))
        top15 = importances.head(15)
        plt.barh(range(len(top15)), top15['Importance'], color='#2ecc71')
        plt.yticks(range(len(top15)), top15['Feature'])
        plt.xlabel('Importance')
        plt.title(f'Importance des variables - {meilleur_nom}', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('ml_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure sauvegardée: ml_feature_importance.png")

# 

print("\n" + "-"*60)
print("13.11 COMPARAISON ML vs MODÈLE POISSON")
print("-"*60)

# Indices pour le Poisson
df_train_poisson = df_model.loc[X_train.index]
df_test_poisson  = df_model.loc[X_test.index]
df_trainval_poisson = df_model.loc[X_train.index.union(X_val.index)]

# Modèle Poisson entraîné sur TRAIN uniquement (comparaison intermédiaire)
formula_poisson_ml = ("Y ~ nb_vivants_centre + C(edu_cat) + C(richesse_cat) + C(region_cat) + "
                      "C(besoin_pf_ns) + C(edu_mari_cat) + C(religion_cat) + "
                      "C(proprietaire_terre_cat) + C(regarde_tv_cat) + C(regarde_journal_cat)")

model_poisson_train = smf.glm(formula=formula_poisson_ml, data=df_train_poisson,
                               family=sm.families.Poisson()).fit(cov_type='HC0')

y_pred_poisson_test = model_poisson_train.predict(df_test_poisson)

rmse_poisson = np.sqrt(mean_squared_error(y_test, y_pred_poisson_test))
mae_poisson  = mean_absolute_error(y_test, y_pred_poisson_test)
r2_poisson   = r2_score(y_test, y_pred_poisson_test)

# Modèle ML réentraîné sur Train+Validation pour comparaison finale équitable
print(f"\n{'='*70}")
print("RÉENTRAÎNEMENT FINAL : Train + Validation")
print(f"{'='*70}")

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

pipeline_final_ml = clone(resultats_ml[meilleur_nom]['pipeline'])
pipeline_final_ml.fit(X_trainval, y_trainval)

y_pred_ml_final = pipeline_final_ml.predict(X_test)
rmse_ml_final = np.sqrt(mean_squared_error(y_test, y_pred_ml_final))
mae_ml_final  = mean_absolute_error(y_test, y_pred_ml_final)
r2_ml_final   = r2_score(y_test, y_pred_ml_final)

# Modèle Poisson réentraîné sur Train+Validation
model_poisson_final = smf.glm(formula=formula_poisson_ml, data=df_trainval_poisson,
                               family=sm.families.Poisson()).fit(cov_type='HC0')
y_pred_poisson_final = model_poisson_final.predict(df_test_poisson)
rmse_poisson_final = np.sqrt(mean_squared_error(y_test, y_pred_poisson_final))
mae_poisson_final  = mean_absolute_error(y_test, y_pred_poisson_final)
r2_poisson_final   = r2_score(y_test, y_pred_poisson_final)

# Tableau comparatif final
print(f"\n{'='*70}")
print("TABLEAU COMPARATIF FINAL: POISSON vs MACHINE LEARNING")
print("     (Les deux modèles réentraînés sur Train+Validation, évalués sur Test)")
print(f"{'='*70}")

comparaison_finale = pd.DataFrame({
    'Modèle': [f'Poisson (robuste HC0)', f'{meilleur_nom} (ML)'],
    'RMSE': [rmse_poisson_final, rmse_ml_final],
    'MAE': [mae_poisson_final, mae_ml_final],
    'R²': [r2_poisson_final, r2_ml_final],
    'N paramètres': [len(model_poisson_final.params), 'Non applicable'],
    'Interprétabilité': ['Élevée (coefficients, p-values)', 'Faible (boîte noire)'],
    'Objectif': ['Inférence causale', 'Prédiction pure']
})

print(comparaison_finale.to_string(index=False))



print("\n" + "-"*60)
print("13.12 FONCTION DE PRÉDICTION ML (MODÈLE FINAL TRAIN+VAL)")
print("-"*60)

def predire_nb_enfants_ml(age=None, edu_cat='Aucune', richesse_cat='Plus pauvre', 
                          region_cat='Adamaoua', nb_vivants_centre=0,
                          besoin_pf_ns='Non_expose', edu_mari_cat='Aucune',
                          religion_cat='Chrétienne', proprietaire_terre_cat='Non',
                          regarde_tv_cat='Pas_du_tout', regarde_journal_cat='Pas_du_tout'):
    """
    Prédit le nombre idéal d'enfants avec le meilleur modèle ML.
    Le modèle final est entraîné sur Train + Validation.
    """
    nouvel_individu = pd.DataFrame({
        'edu_cat': [edu_cat],
        'richesse_cat': [richesse_cat],
        'region_cat': [region_cat],
        'besoin_pf_ns': [besoin_pf_ns],
        'edu_mari_cat': [edu_mari_cat],
        'religion_cat': [religion_cat],
        'proprietaire_terre_cat': [proprietaire_terre_cat],
        'regarde_tv_cat': [regarde_tv_cat],
        'regarde_journal_cat': [regarde_journal_cat],
        'nb_vivants_centre': [nb_vivants_centre]
    })
    
    prediction = pipeline_final_ml.predict(nouvel_individu)[0]
    return round(prediction, 2)

# Exemple
print("\nExemple de prédiction (modèle final):")
pred_exemple = predire_nb_enfants_ml(
    edu_cat='Secondaire',
    richesse_cat='Riche',
    region_cat='Yaoundé',
    nb_vivants_centre=2,
    religion_cat='Musulmane'
)
print(f"Prédiction: {pred_exemple} enfants")

print("\n" + "="*80)
print("SECTION MACHINE LEARNING TERMINÉE")
print("="*80)
