"""
ARBRE DE DECISION — V2
Dataset : 1249 films — dataset augmente (+250)
Prerequis : Data_preprocessing_v2.py doit avoir ete lance avant.

Particularite : on compare 3 normalisations (MinMax, L1, L2) pour trouver
celle qui donne les meilleurs resultats avec l'arbre de decision.
"""

import pandas as pd
import numpy as np
import json, pickle, os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from pathlib import Path

# ----------------------------------------------------------------------------
# CONFIGURATION GENERALE
# ----------------------------------------------------------------------------

VERSION    = 'v2'
LABEL      = '1249 films — dataset augmente (+250)'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data',   VERSION)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', VERSION)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

print(f"\n=== ARBRE DE DECISION {VERSION.upper()} — {LABEL} ===\n")

# ----------------------------------------------------------------------------
# CHARGEMENT DES DONNEES ET DU MAPPING DES GENRES
# ----------------------------------------------------------------------------

csv_file = os.path.join(DATA_DIR, 'movies_processed_enriched.csv')
if not os.path.exists(csv_file):
    print(f"ERREUR : lancez d'abord Data_preprocessing_{VERSION}.py")
    exit(1)

df = pd.read_csv(csv_file)
print(f"{len(df)} films charges")

with open(os.path.join(DATA_DIR, 'genre_mapping.json')) as f:
    genre_mapping = json.load(f)

CLASS_NAMES = list(genre_mapping['major_to_id'].keys())
MAJOR_TO_ID = genre_mapping['major_to_id']

print(f"Classes : {CLASS_NAMES}")
print("\nDistribution :")
for cat, cnt in df['Genre_Major'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"  {cat:15} : {cnt:5} films ({pct:.1f}%)")

# ----------------------------------------------------------------------------
# PREPARATION DES FEATURES ET DES 3 NORMALISATIONS
#
# MinMax : chaque colonne ramenee entre 0 et 1 (deja fait en preprocessing)
# L1     : divise chaque ligne par la somme de ses valeurs (somme ligne = 1.0)
# L2     : divise chaque ligne par sa norme euclidienne (norme ligne = 1.0)
#
# L1 et L2 sont appliquees sur les donnees deja normalisees MinMax
# ----------------------------------------------------------------------------

print("\n--- Preparation des features (MinMax / L1 / L2) ---")

TARGET = 'Genre_Major_ID'
y = df[TARGET].values

BASE_FEATURES = [
    'IMDB_Rating_MM', 'No_of_Votes_MM', 'Released_Year_MM',
    'Custom_Popularity_MM', 'Blockbuster_Score_MM', 'Hidden_Gem_Score_MM',
    'Votes_Log_MM', 'Rating_x_Votes_MM', 'Recent_Popularity_MM', 'High_Quality_MM'
]

X_mm = df[BASE_FEATURES].values
X_l1 = normalize(X_mm, norm='l1')
X_l2 = normalize(X_mm, norm='l2')

NORM_SETS = {'MinMax': X_mm, 'L1': X_l1, 'L2': X_l2}

print(f"  {len(BASE_FEATURES)} features | 3 normalisations : MinMax, L1, L2")
print(f"  Verif L1 ligne 0 — somme = {X_l1[0].sum():.6f}  (doit etre 1.0)")
print(f"  Verif L2 ligne 0 — norme = {np.linalg.norm(X_l2[0]):.6f}  (doit etre 1.0)")

# ----------------------------------------------------------------------------
# GRIDSEARCH : RECHERCHE DES MEILLEURS HYPERPARAMETRES
#
# max_depth        : profondeur max de l'arbre (None = pas de limite)
# min_samples_split: nb minimum de films pour diviser un noeud
# min_samples_leaf : nb minimum de films dans une feuille finale
# criterion        : mesure de qualite des coupures (gini ou entropy)
#
# GridSearchCV teste toutes les combinaisons et retourne la meilleure
# n_jobs=-1 : utilise tous les coeurs du processeur pour accelerer
# ----------------------------------------------------------------------------

print("\n--- Recherche des meilleurs hyperparametres (GridSearchCV) ---")

param_grid = {
    'max_depth':         [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5],
    'criterion':         ['gini', 'entropy']
}

gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
gs.fit(X_mm, y)
best_params = gs.best_params_

print(f"\n  Meilleurs hyperparametres :")
for k, v in best_params.items():
    print(f"    {k:22} = {v}")
print(f"  Accuracy CV (GridSearch) = {gs.best_score_:.4f}")

# ----------------------------------------------------------------------------
# CROSS-VALIDATION PAR NORMALISATION
# On evalue les 3 normalisations avec les meilleurs hyperparametres
# pour determiner laquelle donne les meilleures performances
# ----------------------------------------------------------------------------

print("\n--- Cross-validation (5-fold et 10-fold) par normalisation ---")

cv5      = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)
cv10     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_dt  = DecisionTreeClassifier(**best_params, random_state=42)
norm_results = {}

for nm, X_norm in NORM_SETS.items():
    s5  = cross_val_score(best_dt, X_norm, y, cv=cv5,  scoring='accuracy')
    s10 = cross_val_score(best_dt, X_norm, y, cv=cv10, scoring='accuracy')
    norm_results[nm] = {
        'cv5_mean':    float(s5.mean()),
        'cv5_std':     float(s5.std()),
        'cv5_scores':  [round(float(s), 4) for s in s5],
        'cv10_mean':   float(s10.mean()),
        'cv10_std':    float(s10.std()),
        'cv10_scores': [round(float(s), 4) for s in s10]
    }
    print(f"\n  [{nm}]")
    print(f"    5-fold  -> {s5.mean():.4f} +-{s5.std():.4f}  {[round(s,4) for s in s5]}")
    print(f"    10-fold -> {s10.mean():.4f} +-{s10.std():.4f}  {[round(s,4) for s in s10]}")

# Selection de la meilleure normalisation selon le score 5-fold
best_norm_name = max(norm_results, key=lambda k: norm_results[k]['cv5_mean'])
X_best         = NORM_SETS[best_norm_name]

print(f"\n  Normalisation    CV 5-fold       CV 10-fold   Verdict")
print(f"  {'-'*55}")
for nm, res in norm_results.items():
    tag = "  <- meilleure" if nm == best_norm_name else ""
    print(f"  {nm:<10} {res['cv5_mean']:.4f} +-{res['cv5_std']:.4f}  "
          f"{res['cv10_mean']:.4f} +-{res['cv10_std']:.4f}{tag}")

print(f"\n  Meilleure normalisation : {best_norm_name} "
      f"(CV5 = {norm_results[best_norm_name]['cv5_mean']:.4f})")

# ----------------------------------------------------------------------------
# ENTRAINEMENT FINAL avec les meilleurs hyperparametres et la meilleure norm
# ----------------------------------------------------------------------------

print(f"\n--- Entrainement final — DecisionTree + {best_norm_name} ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_best, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {len(X_train)} films")
print(f"  Test  : {len(X_test)} films")

final_model = DecisionTreeClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(f"  Profondeur : {final_model.get_depth()}")
print(f"  Feuilles   : {final_model.get_n_leaves()}")

# ----------------------------------------------------------------------------
# METRIQUES GLOBALES
# ----------------------------------------------------------------------------

print("\n--- Metriques globales ---")

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test,    y_pred, average='weighted', zero_division=0)
f1  = f1_score(y_test,        y_pred, average='weighted', zero_division=0)

print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision: {pre:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")

# ----------------------------------------------------------------------------
# METRIQUES PAR CATEGORIE
# TP, FP, FN, TN calcules pour chaque genre
# ----------------------------------------------------------------------------

print("\n--- Metriques par categorie ---")

report_dict = classification_report(
    y_test, y_pred, target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)
print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

print(f"  {'Categorie':<18} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8} {'Support':>9}")
print(f"  {'-'*68}")

per_class_metrics = {}
for cat in CLASS_NAMES:
    if cat not in report_dict:
        continue
    r         = report_dict[cat]
    cat_id    = MAJOR_TO_ID[cat]
    mask_true = (y_test == cat_id)
    mask_pred = (y_pred == cat_id)
    tp = ( mask_true &  mask_pred).sum()
    fp = (~mask_true &  mask_pred).sum()
    fn = ( mask_true & ~mask_pred).sum()
    tn = (~mask_true & ~mask_pred).sum()
    cat_acc = (tp + tn) / len(y_test)

    per_class_metrics[cat] = {
        'accuracy':  float(cat_acc),
        'precision': float(r['precision']),
        'recall':    float(r['recall']),
        'f1_score':  float(r['f1-score']),
        'support':   int(r['support']),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }
    print(f"  {cat:<18} {cat_acc:>10.4f} {r['precision']:>11.4f} "
          f"{r['recall']:>8.4f} {r['f1-score']:>8.4f} {int(r['support']):>9}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Matrice de confusion :")
print(f"  {'':18}" + "".join(f"{n:>14}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:<18}" + "".join(f"{v:>14}" for v in row))

best_cat  = max(per_class_metrics, key=lambda k: per_class_metrics[k]['f1_score'])
worst_cat = min(per_class_metrics, key=lambda k: per_class_metrics[k]['f1_score'])
print(f"\n  Meilleure categorie : {best_cat}  (F1={per_class_metrics[best_cat]['f1_score']:.4f})")
print(f"  Pire categorie      : {worst_cat} (F1={per_class_metrics[worst_cat]['f1_score']:.4f})")

# ----------------------------------------------------------------------------
# IMPORTANCE DES FEATURES
# L'arbre calcule automatiquement l'importance de chaque feature
# en mesurant combien elle reduit l'impurete lors des coupures
# La somme de toutes les importances vaut toujours 1.0
# ----------------------------------------------------------------------------

print("\n--- Importance des features ---")

fi = pd.Series(
    final_model.feature_importances_, index=BASE_FEATURES
).sort_values(ascending=False)

for feat, imp in fi.items():
    bar = '|' * int(imp * 50)
    print(f"  {feat:35} : {imp:.4f}  {bar}")

# ----------------------------------------------------------------------------
# COMPARAISON NAIVE BAYES vs ARBRE DE DECISION
# Prerequis : Naive_Bayes_v2.py doit avoir ete lance avant
# ----------------------------------------------------------------------------

print("\n--- Comparaison : Naive Bayes vs Arbre de Decision ---")

nb_path = os.path.join(MODELS_DIR, 'model_info.json')
if os.path.exists(nb_path):
    with open(nb_path) as f:
        nb_info = json.load(f)
    nb_g  = nb_info.get('global_metrics', {})
    nb_pc = nb_info.get('per_class_metrics', {})

    print(f"\n  {'Metrique':<12} {'Naive Bayes':>14} {'Arbre Decision':>16}  Delta")
    print(f"  {'-'*58}")
    for label, (key, dt_val) in {
        'Accuracy':  ('accuracy',  acc),
        'Precision': ('precision', pre),
        'Recall':    ('recall',    rec),
        'F1-Score':  ('f1_score',  f1)
    }.items():
        nb_val = nb_g.get(key, 0)
        d = dt_val - nb_val
        s = '+' if d >= 0 else ''
        w = '<- DT' if d > 0 else ('<- NB' if d < 0 else '=')
        print(f"  {label:<12} {nb_val:>14.4f} {dt_val:>16.4f}  {s}{d:.4f}  {w}")

    print(f"\n  Par categorie :")
    print(f"  {'Categorie':<18} {'NB F1':>8} {'DT F1':>8}  Delta  Meilleur")
    print(f"  {'-'*55}")
    for cat in CLASS_NAMES:
        dt_f1 = per_class_metrics[cat]['f1_score']
        nb_f1 = nb_pc.get(cat, {}).get('f1_score', 0)
        d = dt_f1 - nb_f1
        s = '+' if d >= 0 else ''
        w = 'DT' if d > 0.005 else ('NB' if d < -0.005 else 'Egal')
        print(f"  {cat:<18} {nb_f1:>8.4f} {dt_f1:>8.4f}  {s}{d:.4f}  {w}")
else:
    print(f"  Lancez d'abord Naive_Bayes_{VERSION}.py pour avoir la comparaison")

# ----------------------------------------------------------------------------
# SAUVEGARDE DU MODELE ET DES METRIQUES
# ----------------------------------------------------------------------------

print(f"\n--- Sauvegarde -> models/{VERSION}/ ---")

dt_model_path = os.path.join(MODELS_DIR, 'decision_tree_model.pkl')
with open(dt_model_path, 'wb') as f:
    pickle.dump(final_model, f)

dt_info = {
    'model_type':         'DecisionTreeClassifier',
    'version':            VERSION,
    'label':              LABEL,
    'best_normalization': best_norm_name,
    'features':           BASE_FEATURES,
    'best_params': {
        k: (v if v is not None else 'None')
        for k, v in best_params.items()
    },
    'tree_depth':  int(final_model.get_depth()),
    'n_leaves':    int(final_model.get_n_leaves()),
    'class_names': CLASS_NAMES,
    'normalization_comparison': {
        k: {'cv5': round(v['cv5_mean'], 4), 'cv10': round(v['cv10_mean'], 4)}
        for k, v in norm_results.items()
    },
    'cv_results':      norm_results[best_norm_name],
    'global_metrics': {
        'accuracy':  float(acc),
        'precision': float(pre),
        'recall':    float(rec),
        'f1_score':  float(f1)
    },
    'per_class_metrics': per_class_metrics,
    'feature_importance': fi.to_dict()
}

dt_info_path = os.path.join(MODELS_DIR, 'decision_tree_info.json')
with open(dt_info_path, 'w') as f:
    json.dump(dt_info, f, indent=2)

print(f"  models/{VERSION}/decision_tree_model.pkl")
print(f"  models/{VERSION}/decision_tree_info.json")

print(f"\n=== ARBRE DE DECISION {VERSION.upper()} TERMINE ===")
print(f"  Normalisation : {best_norm_name}")
print(f"  Profondeur    : {final_model.get_depth()}  |  Feuilles : {final_model.get_n_leaves()}")
print(f"  Accuracy      : {acc:.4f}  |  F1 : {f1:.4f}")
print(f"  CV5           : {norm_results[best_norm_name]['cv5_mean']:.4f} "
      f"+-{norm_results[best_norm_name]['cv5_std']:.4f}")