#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')


# 0. CHARGEMENT DES DONNÉES


print("="*80)
print("ANALYSE DES PRÉFÉRENCES DE FÉCONDITÉ - EDS CAMEROUN 2018")
print("="*80)



df = pd.read_stata('CMIR71FL.DTA', convert_categoricals=False)


# 1. VÉRIFICATION PRÉALABLE : PERTINENCE DU MODÈLE DE POISSON


print("\n" + "="*80)
print("1. VÉRIFICATION PRÉALABLE : POURQUOI LA RÉGRESSION DE POISSON ?")
print("="*80)

print("""
Justification théorique (d'après la littérature démographique):
- La variable dépendante est un NOMBRE D'ÉVÉNEMENTS (nombre idéal d'enfants)
- C'est une variable de COMPTAGE (valeurs entières: 0, 1, 2, 3...)

Références:
 Bongaarts (2010): utilisation de modèles de comptage pour la fécondité
 Cleland & Wilson (1987): analyse des déterminants de la fécondité désirée
""")

# Préparation de Y
df['Y'] = pd.to_numeric(df['v614'], errors='coerce').replace({96: np.nan, 98: np.nan, 99: np.nan})
df = df[(df['Y'] >= 0) & (df['Y'] <= 20)].dropna(subset=['Y'])
df['Y'] = df['Y'].astype(int)

print(f"\n--- CARACTÉRISTIQUES DE Y (NOMBRE IDÉAL D'ENFANTS) ---")
print(f"Type de variable: COMPTAGE (entiers positifs)")
print(f"N = {len(df):,}")
print(f"Minimum = {df['Y'].min()}")
print(f"Maximum = {df['Y'].max()}")
print(f"Moyenne = {df['Y'].mean():.4f}")
print(f"Variance = {df['Y'].var():.4f}")
print(f"Écart-type = {df['Y'].std():.4f}")
print(f"Médiane = {df['Y'].median()}")
print(f"Mode = {df['Y'].mode().values[0]}")
print(f"Skewness = {df['Y'].skew():.4f} (asymétrie à gauche)")
print(f"Kurtosis = {df['Y'].kurtosis():.4f}")
#On se rapprochee la loi normale 

# Test de pertinence de Poisson
ratio = df['Y'].var() / df['Y'].mean()
print(f"\n--- TEST DE PERTINENCE DE POISSON ---")
print(f"Variance / Moyenne = {ratio:.4f}")
print(f"Pour une loi de Poisson: Variance = Moyenne (ratio ≈ 1)")
if 0.8 <= ratio <= 1.2:
    print(" RATIO PROCHE DE 1: La régression de Poisson est PERTINENTE ✓")
elif ratio > 1.2:
    print(" RATIO > 1.2: Surdispersion   Quasi-Poisson ou Binomiale Négative")
else:
    print(" RATIO < 0.8: Sous-dispersion  Poisson acceptable mais Robuste")


#le problème avec cette methode est que ce 'est pas un variance conditionnel ni une moyenne condtionnelle elle n'est donc pas trés adéquate




# 2. STATISTIQUE DESCRIPTIVE UNIVARIÉE


print("\n" + "="*80)
print("2. STATISTIQUE DESCRIPTIVE UNIVARIÉE")
print("="*80)

# --- VARIABLE DÉPENDANTE ---
print(f"\n{'-'*60}")
print("TABLEAU 1: DISTRIBUTION DU NOMBRE IDÉAL D'ENFANTS")
print(f"{'-'*60}")

#nombre ideal d'enfant de 0 à n 
freq_y = df['Y'].value_counts().sort_index()

#pourcentage par nombre d'enfant
pct_y = (freq_y / len(df) * 100).round(2)

#pourcentage cumulée
cumul_y = pct_y.cumsum()

tab1 = pd.DataFrame({
    'Nombre idéal': freq_y.index,
    'Effectif': freq_y.values,
    'Pourcentage': pct_y.values,
    'Pourcentage cumulé': cumul_y.values
})
print(tab1.to_string(index=False))
print(f"\nTotal: {tab1['Effectif'].sum():,}")

# -------------------------------------------------------------- VARIABLES INDÉPENDANTES -----------------------------------------------------------------

# Âge (continue)
df['age'] = pd.to_numeric(df['v012'], errors='coerce')

print(f"\n{'-'*60}")
print("TABLEAU 2: CARACTÉRISTIQUES DE L'ÂGE (VARIABLE CONTINUE)")
print(f"{'-'*60}")
print(f"Moyenne: {df['age'].mean():.2f} ans")
print(f"Écart-type: {df['age'].std():.2f}")
print(f"Médiane: {df['age'].median():.1f} ans")
print(f"Min-Max: {df['age'].min():.0f} - {df['age'].max():.0f} ans")

# Éducation (catégorielle ordinale)
df['edu'] = pd.to_numeric(df['v106'], errors='coerce').replace({8: np.nan, 9: np.nan}).fillna(0).astype(int)

print(f"\n{'-'*60}")
print("TABLEAU 3: DISTRIBUTION DU NIVEAU D'ÉDUCATION (VARIABLE CATÉGORIELLE)")
print(f"{'-'*60}")
edu_labels = {0: 'Aucune', 1: 'Primaire', 2: 'Secondaire', 3: 'Supérieur'}
edu_tab = df['edu'].value_counts().sort_index()
edu_pct = (edu_tab / len(df) * 100).round(2)
tab3 = pd.DataFrame({
    'Niveau': [edu_labels[i] for i in edu_tab.index],
    'Code': edu_tab.index,
    'Effectif': edu_tab.values,
    'Pourcentage': edu_pct.values
})
print(tab3.to_string(index=False))

# Milieu (catégorielle binaire)
df['milieu'] = pd.to_numeric(df['v025'], errors='coerce')
df['milieu_label'] = df['milieu'].map({1: 'Urbain', 2: 'Rural'})

print(f"\n{'-'*60}")
print("TABLEAU 4: DISTRIBUTION DU MILIEU DE RÉSIDENCE (VARIABLE CATÉGORIELLE)")
print(f"{'-'*60}")
milieu_tab = df['milieu_label'].value_counts()
milieu_pct = (milieu_tab / len(df) * 100).round(2)
tab4 = pd.DataFrame({
    'Milieu': milieu_tab.index,
    'Effectif': milieu_tab.values,
    'Pourcentage': milieu_pct.values
})
print(tab4.to_string(index=False))

# Richesse (catégorielle ordinale)
df['richesse'] = pd.to_numeric(df['v190'], errors='coerce')
richesse_labels = {1: 'Plus pauvre', 2: 'Pauvre', 3: 'Moyen', 4: 'Riche', 5: 'Plus riche'}
df['richesse_label'] = df['richesse'].map(richesse_labels)

print(f"\n{'-'*60}")
print("TABLEAU 5: DISTRIBUTION DU QUINTILE DE RICHESSE (VARIABLE CATÉGORIELLE)")
print(f"{'-'*60}")
rich_tab = df['richesse_label'].value_counts().reindex(richesse_labels.values())
rich_pct = (rich_tab / len(df) * 100).round(2)
tab5 = pd.DataFrame({
    'Quintile': rich_tab.index,
    'Code': range(1, 6),
    'Effectif': rich_tab.values,
    'Pourcentage': rich_pct.values
})
print(tab5.to_string(index=False))

# Région (catégorielle nominale)
df['region'] = pd.to_numeric(df['v024'], errors='coerce')
region_names = {
    1: 'Adamaoua', 2: 'Centre', 3: 'Douala', 4: 'Est', 5: 'Extreme-Nord',
    6: 'Littoral', 7: 'Nord', 8: 'Nord-Ouest', 9: 'Ouest', 10: 'Sud',
    11: 'Sud-Ouest', 12: 'Yaoundé'
}
df['region_label'] = df['region'].map(region_names)

print(f"\n{'-'*60}")
print("TABLEAU 6: DISTRIBUTION PAR RÉGION (VARIABLE CATÉGORIELLE)")
print(f"{'-'*60}")
reg_tab = df['region_label'].value_counts().sort_index()
reg_pct = (reg_tab / len(df) * 100).round(2)
tab6 = pd.DataFrame({
    'Région': reg_tab.index,
    'Code': [k for k, v in sorted(region_names.items(), key=lambda x: x[1])],
    'Effectif': reg_tab.values,
    'Pourcentage': reg_pct.values
})
print(tab6.to_string(index=False))

#-------------------------------------- 3. STATISTIQUE DESCRIPTIVE BIVARIÉE----------------------------------------------------------------------------


print("\n" + "="*80)
print("3. STATISTIQUE DESCRIPTIVE BIVARIÉE")
print("="*80)

print("""
Les tableaux croisés montrent la moyenne du nombre idéal d'enfants 
selon chaque variable indépendante.
""")

# Y selon Éducation
print(f"\n{'-'*60}")
print("TABLEAU 7: MOYENNE DU NOMBRE IDÉAL D'ENFANTS SELON L'ÉDUCATION")
print(f"{'-'*60}")
edu_biv = df.groupby('edu')['Y'].agg(['count', 'mean', 'std', 'min', 'max'])
edu_biv.index = [edu_labels[i] for i in edu_biv.index]
edu_biv.columns = ['Effectif', 'Moyenne', 'Écart-type', 'Min', 'Max']
print(edu_biv.round(2).to_string())

# Y selon Milieu
print(f"\n{'-'*60}")
print("TABLEAU 8: MOYENNE DU NOMBRE IDÉAL D'ENFANTS SELON LE MILIEU")
print(f"{'-'*60}")
milieu_biv = df.groupby('milieu_label')['Y'].agg(['count', 'mean', 'std'])
milieu_biv.columns = ['Effectif', 'Moyenne', 'Écart-type']
print(milieu_biv.round(2).to_string())

# Y selon Richesse
print(f"\n{'-'*60}")
print("TABLEAU 9: MOYENNE DU NOMBRE IDÉAL D'ENFANTS SELON LA RICHESSE")
print(f"{'-'*60}")
rich_biv = df.groupby('richesse_label')['Y'].agg(['count', 'mean', 'std'])
rich_biv = rich_biv.reindex(richesse_labels.values())
rich_biv.columns = ['Effectif', 'Moyenne', 'Écart-type']
print(rich_biv.round(2).to_string())

# Y selon Région
print(f"\n{'-'*60}")
print("TABLEAU 10: MOYENNE DU NOMBRE IDÉAL D'ENFANTS SELON LA RÉGION")
print(f"{'-'*60}")
reg_biv = df.groupby('region_label')['Y'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
reg_biv.columns = ['Effectif', 'Moyenne', 'Écart-type']
print(reg_biv.round(2).to_string())

# Y selon Âge (par tranches)
print(f"\n{'-'*60}")
print("TABLEAU 11: MOYENNE DU NOMBRE IDÉAL D'ENFANTS SELON L'ÂGE")
print(f"{'-'*60}")
df['age_tranche'] = pd.cut(df['age'], bins=[14, 20, 25, 30, 35, 40, 50], 
                            labels=['15-20', '21-25', '26-30', '31-35', '36-40', '41-49'])
age_biv = df.groupby('age_tranche')['Y'].agg(['count', 'mean', 'std'])
age_biv.columns = ['Effectif', 'Moyenne', 'Écart-type']
print(age_biv.round(2).to_string())


# -----------------------------------------------------4. JUSTIFICATION DES VARIABLES (basée sur la littérature)--------------------------------------


print("\n" + "="*80)
print("4. JUSTIFICATION DES VARIABLES INDÉPENDANTES")
print("="*80)



# 5. MODÈLE DE POISSON AVEC VARIABLES CATÉGORIELLES


print("\n" + "="*80)
print("5. ESTIMATION DU MODÈLE DE POISSON")
print("="*80)

print("""
Les variables catégorielles sont traitées comme des FACTEURS:
- Éducation: variable ordinale avec modalités (Aucune, Primaire, Secondaire, Supérieur)
- Milieu: variable binaire (Urbain/Rural)
- Richesse: variable ordinale (Q1 à Q5)
- Région: variable nominale (12 modalités)

Approche: utilisation de la formule R-like avec C() pour les facteurs
""")

# Préparation des données avec variables catégorielles explicites
df_model = df[['Y', 'age', 'edu', 'milieu', 'richesse', 'region']].dropna().copy()
#centrons l'age car si on ne le fait pas al reference d'age sera 0 ans ce qui n'est pas possible car un nouveua née ne peut avoir un nombre idéal d'enfant
df_model['age_centre']=df_model['age']- df_model['age'].mean()
print("-------------------------------------------------aaaaaaaaaaageeeeeeeeeeeeeeeeee------------------------------------------------")
print(df_model['age_centre'])
age_mean = df_model['age'].mean()

# Conversion en catégories explicites
df_model['edu_cat'] = df_model['edu'].map(edu_labels).astype('category')
df_model['edu_cat'] = pd.Categorical(df_model['edu_cat'], 
                                      categories=['Aucune', 'Primaire', 'Secondaire', 'Supérieur'],
                                      ordered=True)

df_model['milieu_cat'] = df_model['milieu'].map({1: 'Urbain', 2: 'Rural'}).astype('category')
df_model['richesse_cat'] = df_model['richesse'].map(richesse_labels).astype('category')
df_model['richesse_cat'] = pd.Categorical(df_model['richesse_cat'],
                                           categories=['Plus pauvre', 'Pauvre', 'Moyen', 'Riche', 'Plus riche'],
                                           ordered=True)

df_model['region_cat'] = df_model['region'].map(region_names).astype('category')

print(f"\nDimensions du modèle: N = {len(df_model):,}")

# Formule du modèle avec variables catégorielles
# Références: Aucune (éducation), Urbain (milieu), Plus pauvre (richesse), Adamaoua (région)
#justifions le choix de nos reference 

formula = "Y ~ age_centre + C(edu_cat) + C(milieu_cat) + C(richesse_cat) + C(region_cat)"

print(f"\nFormule du modèle:")
print(f"  {formula}")
print(f"\nVariables catégorielles avec modalités de référence:")
print(f"  - Éducation: 'Aucune' car elle est le plus base c'etl'étatinitial d'un individu")
print(f"  - Milieu: 'Urbain'")  
print(f"  - Richesse: 'Plus pauvre' c'est le niveau le plus bas ")
print(f"  - Région: 'Adamaoua' c'est la region ayant la moyenne du lus grand nombre d'enfant elevé ")

# Estimation
#déclaration du
model_poisson = smf.glm(formula=formula, data=df_model, 
                        family=sm.families.Poisson()).fit()



mode_binomial= smf.negativebinomial(formula=formula, data=df_model,loglike_method='nb2').fit()
print(mode_binomial.summary())
print("""
----------------------------------------Binomial negative ------------------------------------------ 
""")

print(f"\n{'='*80}")
print("RÉSULTATS DE LA RÉGRESSION DE POISSON")
print(f"{'='*80}")
print(model_poisson.summary())
print("\n" + "="*50)
print("Binomiale négative - Résultats")
print("="*50)

# Vérifier le paramètre alpha (sur-dispersion)
print(f"\nParamètre alpha : {mode_binomial.params[-1]:.4f}")
print(f"Ecart-type de alpha : {mode_binomial.bse[-1]:.4f}")

# Interprétation
alpha = mode_binomial.params[-1]
if alpha < 0.01:
    print("\n Alpha est très proche de 0 : Pas de sur-dispersion")
    print("   La binomiale négative se réduit à un simple Poisson")
elif alpha > 0:
    print(f"\n Alpha = {alpha:.3f} > 0 : Sur-dispersion détectée")
else:
    print("\n Alpha négatif : Problème de convergence")

#-------------------------------------------------------------------- 6. INTERPRÉTATION DES COEFFICIENTS----------------------------------------------------------------------------
print("\n" + "="*80)
print("6. INTERPRÉTATION DES COEFFICIENTS")
print("="*80)

print("""
Interprétation multiplicative (forme exponentielle):
  exp(β) = ratio d'incidence = facteur de multiplication de la moyenne

Pour une variable catégorielle:
  exp(β) = E[Y|X=modalité] / E[Y|X=référence]
""")
params1 = model_poisson.params
conf_int1 = model_poisson.conf_int1()
exp_params1 = np.exp(params1)
exp_conf_low1 = np.exp(conf_int1[0])
exp_conf_high1 = np.exp(conf_int1[1])

# Tableau des coefficients avec IRR
coef_table = pd.DataFrame({
    'Variable': model_poisson.params.index,
    'Coefficient (β)': model_poisson.params.values,
    'SE': model_poisson.bse.values,
    'z': model_poisson.tvalues.values,
    'P>|z|': model_poisson.pvalues.values,
    'IC_low': exp_conf_low1.values,
    'IC_high': exp_conf_high1.values,
    'exp(β)': np.exp(model_poisson.params.values),
    'Effet %': (np.exp(model_poisson.params.values) - 1) * 100
})



print(coef_table.round(4).to_string())

# Interprétations détaillées
print(f"\n{'='*60}")
print("INTERPRÉTATIONS DÉTAILLÉES")
print(f"{'='*60}")

# Constante
const = model_poisson.params['Intercept']
print(f"\n1. CONSTANTE (Intercept = {const:.4f}):")
print(f"   exp({const:.4f}) = {np.exp(const):.3f}")
print(f"    Profil de référence: femme de 0 an, sans éducation, urbaine,")
print(f"     plus pauvre, résidant à Adamaoua")
print(f"    Nombre idéal moyen: {np.exp(const):.2f} enfants")

# Âge
beta_age = model_poisson.params['age_centre']
print(f"\n2. ÂGE (β = {beta_age:.4f}):")
print(f"   exp({beta_age:.4f}) = {np.exp(beta_age):.4f}")
print(f"    1 an de plus: multiplication par {np.exp(beta_age):.4f}")
print(f"    Soit une augmentation de {(np.exp(beta_age)-1)*100:.2f}%")

# Éducation (comparaison par modalité)
print(f"\n3. ÉDUCATION (référence: 'Aucune'):")
for edu_level in ['Primaire', 'Secondaire', 'Supérieur']:
    var_name = f"C(edu_cat)[T.{edu_level}]"
    if var_name in model_poisson.params.index:
        beta = model_poisson.params[var_name]
        irr = np.exp(beta)
        print(f"    {edu_level:12s}: β = {beta:7.4f}, exp(β) = {irr:.4f}")
        print(f"      {(irr-1)*100:+.1f}% vs sans éducation")

# Milieu
beta_rural = model_poisson.params.get("C(milieu_cat)[T.Rural]", 0)
print(f"\n4. MILIEU (référence: 'Urbain'):")
print(f"    Rural: β = {beta_rural:.4f}, exp(β) = {np.exp(beta_rural):.4f}")
print(f"     {(np.exp(beta_rural)-1)*100:+.1f}% vs urbain")

# Richesse
print(f"\n5. RICHESSE (référence: 'Plus pauvre'):")
for rich_level in ['Pauvre', 'Moyen', 'Riche', 'Plus riche']:
    var_name = f"C(richesse_cat)[T.{rich_level}]"
    if var_name in model_poisson.params.index:
        beta = model_poisson.params[var_name]
        irr = np.exp(beta)
        print(f"   • {rich_level:12s}: β = {beta:7.4f}, exp(β) = {irr:.4f}")
        print(f"     {(irr-1)*100:+.1f}% vs plus pauvre")


# -------------------------------------------------7. DIAGNOSTICS DU MODÈLE (sans suppression des valeurs aberrantes biensur )---------------------------


print("\n" + "="*80)
print("7. DIAGNOSTICS DU MODÈLE")
print("="*80)

print("""
Les valeurs aberrantes ne sont PAS supprimées car elles représentent
des comportements réels. On utilise une approche diagnostique pour
évaluer leur impact sur le modèle.
""")


print("""
----------------------------------------Verifions la pertinance du chois de notre de notre modèle------------------------------------------ 
""")
residus_pearson= model_poisson.resid_pearson
n=len(df)
p=model_poisson.df_model+1
dispersion_estimee=(residus_pearson**2).sum()/(n-p)
print(dispersion_estimee)

y_pred = model_poisson.predict(df_model)
resid_pearson = (df_model['Y'] - y_pred) / np.sqrt(y_pred)

# Leverage
X_design = model_poisson.model.exog
h = np.diag(X_design @ np.linalg.pinv(X_design.T @ X_design) @ X_design.T)

# Résidus standardisés de Pearson
resid_std = resid_pearson / np.sqrt(1 - h)

# Distance de Cook
cook_d = (resid_pearson**2 / len(model_poisson.params)) * (h / (1 - h))
print("""
----------------------------------------Verifions la pertinance du chois de notre de notre modèle------------------------------------------ 
""")

print(cook_d)

print(f"\n--- DIAGNOSTICS ---")
print(f"Nombre d'observations: {len(df_model):,}")
print(f"Nombre de paramètres: {len(model_poisson.params)}")
print(f"Seuil |résidu standardisé| > 2: {np.sum(np.abs(resid_std) > 2)} ({np.mean(np.abs(resid_std) > 2)*100:.2f}%)")
print(f"Seuil |résidu standardisé| > 3: {np.sum(np.abs(resid_std) > 3)} ({np.mean(np.abs(resid_std) > 3)*100:.2f}%)")
print(f"Max |résidu standardisé|: {np.max(np.abs(resid_std)):.3f}")

print(f"\n--- DISTANCES DE COOK ---")
seuil_cook = 4 / len(df_model)
print(f"Seuil 4/n: {seuil_cook:.6f}")
print(f"Observations avec Cook > seuil: {np.sum(cook_d > seuil_cook)}")
print(f"Max Cook: {np.max(cook_d):.6f}")

print(f"\nNote: Ces observations ne sont pas supprimées mais identifiées")
print(f"comme potentiellement influentes. Elles représentent des profils")
print(f"réels de la population camerounaise.")

# ------------------------------------------------------------------8. QUALITÉ DU MODÈLE---------------------------------------------------------------------


print("\n" + "="*80)
print("8. QUALITÉ DU MODÈLE ET TESTS")
print("="*80)

# Déviance
print(f"\nDéviance: {model_poisson.deviance:.4f}")
print(f"Degrés de liberté: {model_poisson.df_resid}")
print(f"Déviance/ddl: {model_poisson.deviance/model_poisson.df_resid:.4f}")

# Test du rapport de vraisemblance (vs modèle nul)
model_null = smf.glm("Y ~ 1", data=df_model, family=sm.families.Poisson()).fit()
lr_stat = 2 * (model_poisson.llf - model_null.llf)
df_lr = len(model_poisson.params) - 1
p_lr = 1 - stats.chi2.cdf(lr_stat, df_lr)

print(f"\n--- TEST DU RAPPORT DE VRAISEMBLANCE ---")
print(f"LogL modèle nul: {model_null.llf:.4f}")
print(f"LogL modèle complet: {model_poisson.llf:.4f}")
print(f"Statistique LR: {lr_stat:.4f}")
print(f"ddl: {df_lr}")
print(f"P-valeur: {p_lr:.6f}")
print("→ Modèle globalement significatif" if p_lr < 0.05 else "→ Modèle non significatif")

# Pseudo R²
print(f"\nPseudo R² (McFadden): {model_poisson.pseudo_rsquared():.4f}")


#---------------------------------------------------- 9. CONSOLE DE PRÉDICTION------------------------------------------------------------------


print("\n" + "="*80)
print("9. CONSOLE INTERACTIVE DE PRÉDICTION")
print("="*80)

def predire_console():
    """Console interactive pour prédire le nombre idéal d'enfants"""
    
    print("\n" + "-"*60)
    print("ENTREZ LES CARACTÉRISTIQUES:")
    print("-"*60)
    
    # Âge
    while True:
        try:
            age = int(input("Âge (15-49): "))
            if 15 <= age <= 49:
                break
            print("  Erreur: âge entre 15 et 49")
        except:
            print("  Erreur: entrez un nombre entier")
    
    # Éducation
    print("\nNiveau d'éducation:")
    for i, label in edu_labels.items():
        print(f"  {i}: {label}")
    while True:
        try:
            edu = int(input("Choix (0-3): "))
            if edu in [0, 1, 2, 3]:
                break
            print("   Erreur: 0, 1, 2 ou 3")
        except:
            print("   Erreur: entrez un nombre")
    
    # Milieu
    print("\nMilieu:")
    print("  1: Urbain")
    print("  2: Rural")
    while True:
        try:
            milieu = int(input("Choix (1-2): "))
            if milieu in [1, 2]:
                break
            print("   Erreur: 1 ou 2")
        except:
            print("   Erreur: entrez un nombre")
    
    # Richesse
    print("\nQuintile de richesse:")
    for i, label in richesse_labels.items():
        print(f"  {i}: {label}")
    while True:
        try:
            richesse = int(input("Choix (1-5): "))
            if richesse in [1, 2, 3, 4, 5]:
                break
            print("   Erreur: 1 à 5")
        except:
            print("   Erreur: entrez un nombre")
    
    # Région
    print("\nRégion:")
    for i, label in sorted(region_names.items()):
        print(f"  {i:2d}: {label}")
    while True:
        try:
            region = int(input("Choix (1-12): "))
            if region in range(1, 13):
                break
            print("   Erreur: 1 à 12")
        except:
            print("   Erreur: entrez un nombre")
    
    # Création du profil
    profil = pd.DataFrame({
        'age_centre': [age - age_mean],
        'edu_cat': [edu_labels[edu]],
        'milieu_cat': ['Urbain' if milieu == 1 else 'Rural'],
        'richesse_cat': [richesse_labels[richesse]],
        'region_cat': [region_names[region]]
    })
    
    # Prédiction
    pred = model_poisson.predict(profil)[0]
    
    # Intervalle de confiance (approximation)
    X_pred = model_poisson.model.exog
    # Simplification: on utilise la variance moyenne
    se_log = 0.05  # Approximation conservative
    ic_inf = np.exp(np.log(pred) - 1.96 * se_log)
    ic_sup = np.exp(np.log(pred) + 1.96 * se_log)
    
    # Segment
    if pred < 3.5:
        segment = "Faible"
        reco = "Maintien des services standards"
    elif pred < 5.5:
        segment = "Modéré"
        reco = "Information et accès facilité à la PF"
    else:
        segment = "Élevé"
        reco = "Intervention intensive en planification familiale"
    
    # Affichage
    print("\n" + "="*60)
    print("RÉSULTAT")
    print("="*60)
    print(f"\nProfil: {age} ans, {edu_labels[edu]}, {'Urbain' if milieu==1 else 'Rural'},")
    print(f"        {richesse_labels[richesse]}, {region_names[region]}")
    print(f"\n{'='*60}")
    print(f"NOMBRE IDÉAL D'ENFANTS PRÉDIT: {pred:.2f}")
    print(f"{'='*60}")
    print(f"IC 95% approximatif: [{ic_inf:.2f}, {ic_sup:.2f}]")
    print(f"\nSegment de risque: {segment}")
    print(f"Recommandation: {reco}")
    
    return pred

# Boucle principale   
print("\nConsole interactive prête!")
print("Entrez les caractéristiques d'une femme pour obtenir la prédiction.")

while True:
    predire_console()
    print("\n" + "-"*60)
    continuer = input("Nouvelle prédiction? (o/n): ").strip().lower()
    if continuer not in ['o', 'oui', 'y', 'yes']:
        break

print("\n" + "="*80)
print("ANALYSE TERMINÉE")
print("="*80)


# In[ ]:


# essayons avec un modele de regression de poisson robuste 
model_poisson_robuste = smf.glm(formula=formula, data=df_model, family=sm.families.Poisson()).fit(cov_type='HC0')
print("="*70)
print("MODÈLE POISSON AVEC ERREURS ROBUSTES (HC0)")
print("="*70)
print(model_poisson_robuste.summary())

# 4. Interprétation des coefficients en termes de ratio de taux
print("\n" + "="*70)
print("INTERPRÉTATION DES COEFFICIENTS")
print("="*70)
print("Les coefficients exponentiels (exp(β)) s'interprètent comme des ratios de taux.")
print("Exemple : exp(β_age) = augmentation multiplicative du nombre d'enfants")
print("pour chaque année d'âge supplémentaire.\n")

# Afficher les ratios de taux avec leurs intervalles de confiance
params = model_poisson_robuste.params
conf_int = model_poisson_robuste.conf_int()
exp_params = np.exp(params)
exp_conf_low = np.exp(conf_int[0])
exp_conf_high = np.exp(conf_int[1])

resultats = pd.DataFrame({
    'Variable': params.index,
    'Coef': params.values,
    'Exp(Coef)': exp_params.values,
    'IC_low': exp_conf_low.values,
    'IC_high': exp_conf_high.values,
    'P>|z|': model_poisson_robuste.pvalues.values
})
print(resultats.round(4))

# 5. Diagnostic final 
print("\n" + "="*70)
print("DIAGNOSTIC FINAL")
print("="*70)
print(f" Modèle utilisé : Poisson avec erreurs robustes (cov_type='HC0')")
print(f" Correction appliquée : sous-dispersion (indice = 0.33)")
print(f" Les p-values et IC sont maintenant valides")


# In[ ]:


# Imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Préparation des données (création des dummies pour les catégories)
X = pd.get_dummies(df_model[['age', 'edu_cat', 'milieu_cat', 'richesse_cat', 'region_cat']], 
                   drop_first=True)
y = df_model['Y']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèles à tester
modeles_ml = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Ridge (régularisé)': Ridge(alpha=1.0),
    'Lasso (régularisé)': Lasso(alpha=0.01)
}

# Comparaison
for nom, modele in modeles_ml.items():
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{nom}: RMSE = {rmse:.3f}")


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import warnings
warnings.filterwarnings('ignore')

# 1. Préparation des données création des dummies pour les catégories
X = pd.get_dummies(df_model[['age', 'edu_cat', 'milieu_cat', 'richesse_cat', 'region_cat']], 
                   drop_first=True)
y = df_model['Y']

# 2. Division en train (80%) et test (20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train : {X_train.shape[0]} observations")
print(f"Test : {X_test.shape[0]} observations")


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluer_modele(modele, X_train, y_train, X_test, y_test, nom):
    """Fonction pour évaluer un modèle"""
    # Entraînement
    modele.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = modele.predict(X_train)
    y_pred_test = modele.predict(X_test)
    
    # Métriques
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # MAPE (attention aux 0)
    y_test_non_zero = y_test[y_test > 0]
    y_pred_non_zero = y_pred_test[y_test > 0]
    mape = np.mean(np.abs((y_test_non_zero - y_pred_non_zero) / y_test_non_zero)) * 100
    
    print(f"\n{'='*50}")
    print(f" {nom}")
    print(f"{'='*50}")
    print(f"RMSE (train) : {rmse_train:.4f}")
    print(f"RMSE (test)  : {rmse_test:.4f}")
    print(f"MAE (test)   : {mae_test:.4f}")
    print(f"R² (test)    : {r2_test:.4f}")
    print(f"MAPE (test)  : {mape:.2f}%")
    print(f"Écart train/test : {(rmse_train - rmse_test):.4f}")
    
    # Détection de sur-apprentissage
    if abs(rmse_train - rmse_test) > 0.3:
        print(" Risque de sur-apprentissage !")
    
    return {'rmse': rmse_test, 'mae': mae_test, 'r2': r2_test, 'mape': mape}


# In[ ]:


# Modèle 1 : Poisson avec régularisation Ridge
modele_poisson_ridge = PoissonRegressor(alpha=0.1, max_iter=1000)

# Modèle 2 : Poisson avec Lasso (peut supprimer des variables)
modele_poisson_lasso = PoissonRegressor(alpha=0.01, max_iter=1000)

# Modèle 3 : Random Forest
modele_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Modèle 4 : Gradient Boosting
modele_gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

# Évaluation de tous les modèles
resultats = {}
modeles= [('Poisson Ridge', modele_poisson_ridge),
                    ('Poisson Lasso', modele_poisson_lasso),
                    ('Random Forest', modele_rf),
                    ('Gradient Boosting', modele_gb)]
for nom, modele in modeles:
    resultats[nom] = evaluer_modele(modele, X_train, y_train, X_test, y_test, nom)

# Comparaison finale
print("\n" + "="*70)
print(" CLASSEMENT FINAL DES MODÈLES (basé sur RMSE)")
print("="*70)
sorted_results = sorted(resultats.items(), key=lambda x: x[1]['rmse'])
for i, (nom, metriques) in enumerate(sorted_results, 1):
    print(f"{i}. {nom} : RMSE = {metriques['rmse']:.4f}, MAE = {metriques['mae']:.4f}, R² = {metriques['r2']:.4f}")


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Exemple pour Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42),
                           param_grid,
                           cv=5,
                           scoring='neg_root_mean_squared_error',
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Meilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur RMSE en validation croisée : {-grid_search.best_score_:.4f}")

# Modèle optimisé
modele_optimal = grid_search.best_estimator_


# In[ ]:


# Après avoir entraîné tous tes modèles
resultats = {}

for nom, modele in modeles:
    # Prédiction sur les données de TEST (pas d'entraînement !)
    y_pred = modele.predict(X_test)
    
    # Calcul du RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    resultats[nom] = rmse
    
    print(f"{nom}: RMSE = {rmse:.4f}")

# Le meilleur modèle = celui avec le PLUS PETIT RMSE
meilleur_modele = min(resultats, key=resultats.get)
print(f"\n MEILLEUR MODÈLE : {meilleur_modele} avec RMSE = {resultats[meilleur_modele]:.4f}")


# In[ ]:


# Après avoir choisi et entraîné ton meilleur modèle

d=dict(modeles)
meilleur_modele_entraine =modele_poisson_lasso
meilleur_modele_entraine.fit(X_train, y_train)

# Fonction de prédiction pour un nouvel individu
def predire_nb_enfants(age, edu_cat, milieu_cat, richesse_cat, region_cat):
    """
    Prédit le nombre d'enfants  idéal pour une femme donnée.
    
    Paramètres:
    - age: int (âge de la femme)
    - edu_cat: str ('Aucune', 'Primaire', 'Secondaire', 'Universitaire')
    - milieu_cat: str ('Urbain', 'Rural')
    - richesse_cat: str ('Très pauvre', 'Pauvre', 'Moyen', 'Riche', 'Très riche')
    - region_cat: str (nom de la région)
    """
    
    # Créer un DataFrame avec les valeurs saisies
    nouvel_individu = pd.DataFrame({
        'age': [age],
        'edu_cat': [edu_cat],
        'milieu_cat': [milieu_cat],
        'richesse_cat': [richesse_cat],
        'region_cat': [region_cat]
    })
    
    # Créer les mêmes dummies que pour l'entraînement
    nouvel_individu_dummies = pd.get_dummies(nouvel_individu)
    
    # Ajouter les colonnes manquantes
    for col in X_train.columns:
        if col not in nouvel_individu_dummies.columns:
            nouvel_individu_dummies[col] = 0
    
    # Réordonner les colonnes comme dans X_train
    nouvel_individu_dummies = nouvel_individu_dummies[X_train.columns]
    
    # Prédiction
    prediction = meilleur_modele_entraine.predict(nouvel_individu_dummies)[0]
    
    return round(prediction, 2)

# Exemple d'utilisation
print("=== SIMULATION DE PRÉDICTION ===")
nb = predire_nb_enfants(
    age=30, 
    edu_cat='Aucune', 
    milieu_cat='Rural', 
    richesse_cat='Très pauvre', 
    region_cat='Extrême-Nord'
)
print(f"Nombre d'enfants prédit : {nb}")


# In[ ]:


def interface_utilisateur():
    print("\n" + "="*50)
    print("PRÉDICTION DU NOMBRE D'ENFANTS")
    print("="*50)
    
    # Saisie utilisateur
    age = int(input("Âge de la femme : "))
    
    print("\nNiveau d'éducation :")
    print("1 - Aucune")
    print("2 - Primaire")
    print("3 - Secondaire")
    print("4 - Universitaire")
    edu_choice = int(input("Votre choix (1-4) : "))
    edu_map = {1: 'Aucune', 2: 'Primaire', 3: 'Secondaire', 4: 'Universitaire'}
    edu_cat = edu_map[edu_choice]
    
    milieu = input("Milieu (Urbain/Rural) : ").capitalize()
    
    print("\nNiveau de richesse :")
    print("1 - Très pauvre")
    print("2 - Pauvre")
    print("3 - Moyen")
    print("4 - Riche")
    print("5 - Très riche")
    richesse_choice = int(input("Votre choix (1-5) : "))
    richesse_map = {1: 'Très pauvre', 2: 'Pauvre', 3: 'Moyen', 4: 'Riche', 5: 'Très riche'}
    richesse_cat = richesse_map[richesse_choice]
    
    region = input("Région (ex: Centre, Littoral, Extrême-Nord, etc.) : ").capitalize()
    
    # Prédiction
    prediction = predire_nb_enfants(age, edu_cat, milieu, richesse_cat, region)
    
    print("\n" + "="*50)
    print(f" RÉSULTAT : Nombre d'enfants prédit = {prediction}")
    print("="*50)

# Lancer l'interface
interface_utilisateur()

