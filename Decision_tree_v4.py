"""
🌳 ARBRE DE DÉCISION — V4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADAPTATIONS PAR RAPPORT À V3 :

  FEATURES :
  ❌ V3 : 10 features corrélées (corr jusqu'à 1.00)
  ✅ V4 : 4 features indépendantes issues de preprocessing_v4 :
          → IMDB_Rating_MM   (normalisée MinMax)
          → Votes_Log_MM     (normalisée MinMax)
          → Movie_Age_MM     (normalisée MinMax)
          → Is_Hidden_Gem    (binaire 0/1, non normalisée)

  NORMALISATION L1 / L2 :
  ❌ V3 : appliquait L1/L2 sur 10 features déjà normalisées → inutile
  ✅ V4 : L1/L2 appliquées uniquement sur les 3 features continues MM
          Is_Hidden_Gem binaire exclue de L1/L2 puis réintégrée

  FICHIERS :
  ❌ V3 : movies_processed_enriched.csv + genre_mapping.json
  ✅ V4 : movies_processed_v4.csv       + genre_mapping_v4.json

  CLASSES :
  ❌ V3 : Drama = 43.8% → biais
  ✅ V4 : classes équilibrées par undersampling (preprocessing_v4)

Prérequis : Data_preprocessing_v4.py doit avoir été lancé avant.
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

VERSION    = 'v4'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data',   VERSION)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', VERSION)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# ── Features V4 ─────────────────────────────────────────────────────────────
# 3 features continues (seront soumises à L1/L2) + 1 binaire (exclue)
CONTINUOUS_FEATURES = ['IMDB_Rating_MM', 'Votes_Log_MM', 'Movie_Age_MM']
BINARY_FEATURES     = ['Is_Hidden_Gem']
ALL_FEATURES        = CONTINUOUS_FEATURES + BINARY_FEATURES

print("\n" + "🌳"*40)
print("  ARBRE DE DÉCISION V4 — Features optimisées + Normalisation corrigée")
print("🌳"*40 + "\n")

# ══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════
print("="*70)
print("📥 CHARGEMENT")
print("="*70)

csv_file     = os.path.join(DATA_DIR, 'movies_processed_v4.csv')
mapping_file = os.path.join(DATA_DIR, 'genre_mapping_v4.json')

if not os.path.exists(csv_file):
    print(f"❌ ERREUR : fichier introuvable → {csv_file}")
    print(f"   Lance d'abord : python Data_preprocessing_v4.py")
    exit(1)

df = pd.read_csv(csv_file)
print(f"  ✅ {len(df)} films chargés depuis movies_processed_v4.csv")

with open(mapping_file) as f:
    genre_mapping = json.load(f)

MAJOR_TO_ID = genre_mapping['major_to_id']
ID_TO_MAJOR = {int(k): v for k, v in genre_mapping['id_to_major'].items()}
CLASS_NAMES = list(MAJOR_TO_ID.keys())

print(f"\n  Classes : {CLASS_NAMES}")
print(f"\n  Distribution (équilibrée par preprocessing_v4) :")
for cat, cnt in df['Genre_Major'].value_counts().items():
    pct = cnt / len(df) * 100
    bar = '█' * (cnt // 30)
    print(f"    {cat:15} : {cnt:5} ({pct:.1f}%)  {bar}")

# Vérification features
missing = [f for f in ALL_FEATURES if f not in df.columns]
if missing:
    print(f"\n❌ Features manquantes : {missing}")
    print(f"   Relance Data_preprocessing_v4.py")
    exit(1)

print(f"\n  Features continues ({len(CONTINUOUS_FEATURES)}) : {CONTINUOUS_FEATURES}")
print(f"  Features binaires  ({len(BINARY_FEATURES)})     : {BINARY_FEATURES}")

# ══════════════════════════════════════════════════════════════════════════
# 2. PRÉPARATION X / y ET NORMALISATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📐 PRÉPARATION FEATURES ET NORMALISATIONS")
print("="*70)

y = df['Genre_Major_ID'].values

# Matrices de base
X_cont   = df[CONTINUOUS_FEATURES].values   # 3 features continues
X_binary = df[BINARY_FEATURES].values       # 1 feature binaire

# MinMax : déjà fait en preprocessing → on réutilise directement
X_mm = df[ALL_FEATURES].values

# L1 et L2 : appliquées SEULEMENT sur les 3 features continues
# puis on réintègre Is_Hidden_Gem sans la normaliser
# Raison : normaliser un binaire 0/1 n'a aucun sens mathématique
X_cont_l1 = normalize(X_cont, norm='l1')
X_cont_l2 = normalize(X_cont, norm='l2')

X_l1 = np.hstack([X_cont_l1, X_binary])   # 3 features L1 + binaire brute
X_l2 = np.hstack([X_cont_l2, X_binary])   # 3 features L2 + binaire brute

NORM_SETS = {
    'MinMax': X_mm,
    'L1':     X_l1,
    'L2':     X_l2
}

print(f"\n  MinMax : features déjà normalisées par preprocessing_v4")
print(f"  L1     : norm L1 sur {CONTINUOUS_FEATURES}")
print(f"           Is_Hidden_Gem réintégrée sans normalisation")
print(f"  L2     : norm L2 sur {CONTINUOUS_FEATURES}")
print(f"           Is_Hidden_Gem réintégrée sans normalisation")
print(f"\n  Vérif L1 ligne 0 — somme features continues = {X_cont_l1[0].sum():.6f}")
print(f"  Vérif L2 ligne 0 — norme features continues = {np.linalg.norm(X_cont_l2[0]):.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. GRIDSEARCH HYPERPARAMÈTRES (sur MinMax)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("⚙️  OPTIMISATION HYPERPARAMÈTRES (GridSearchCV sur MinMax)")
print("="*70)

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

print(f"\n  Meilleurs hyperparamètres :")
for k, v in best_params.items():
    print(f"    {k:22} = {v}")
print(f"  Accuracy CV GridSearch = {gs.best_score_:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION MinMax / L1 / L2
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📐 CROSS-VALIDATION 5-fold & 10-fold PAR NORMALISATION")
print("="*70)

cv5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_dt = DecisionTreeClassifier(**best_params, random_state=42)

norm_results = {}
for nm, X_norm in NORM_SETS.items():
    s5  = cross_val_score(best_dt, X_norm, y, cv=cv5,  scoring='accuracy')
    s10 = cross_val_score(best_dt, X_norm, y, cv=cv10, scoring='accuracy')
    norm_results[nm] = {
        'cv5_mean':   float(s5.mean()),
        'cv5_std':    float(s5.std()),
        'cv5_scores': [round(float(s), 4) for s in s5],
        'cv10_mean':  float(s10.mean()),
        'cv10_std':   float(s10.std()),
        'cv10_scores':[round(float(s), 4) for s in s10],
    }
    print(f"\n  [{nm}]")
    print(f"    5-fold  → {s5.mean():.4f} ±{s5.std():.4f}  {[round(s,4) for s in s5]}")
    print(f"    10-fold → {s10.mean():.4f} ±{s10.std():.4f}  {[round(s,4) for s in s10]}")

best_norm_name = max(norm_results, key=lambda k: norm_results[k]['cv5_mean'])
X_best = NORM_SETS[best_norm_name]

print(f"\n  {'Normalisation':<10} {'CV 5-fold':>14} {'CV 10-fold':>14}  Verdict")
print(f"  {'-'*55}")
for nm, res in norm_results.items():
    tag = "  🏆" if nm == best_norm_name else ""
    print(f"  {nm:<10} {res['cv5_mean']:.4f} ±{res['cv5_std']:.4f}"
          f"  {res['cv10_mean']:.4f} ±{res['cv10_std']:.4f}{tag}")

print(f"\n  Meilleure normalisation : {best_norm_name}"
      f" (CV5 = {norm_results[best_norm_name]['cv5_mean']:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 5. ENTRAÎNEMENT FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"🎓 ENTRAÎNEMENT FINAL — DecisionTree + {best_norm_name}")
print("="*70)

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

# ══════════════════════════════════════════════════════════════════════════
# 6. MÉTRIQUES GLOBALES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📈 MÉTRIQUES GLOBALES")
print("="*70)

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test,    y_pred, average='weighted', zero_division=0)
f1  = f1_score(y_test,        y_pred, average='weighted', zero_division=0)

print(f"\n  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision: {pre:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")

print(f"\n  Comparaison V3 → V4 :")
print(f"    Accuracy V3 : (voir models/v3/decision_tree_info.json)")
print(f"    Accuracy V4 : {acc*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════
# 7. MÉTRIQUES PAR CATÉGORIE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📊 MÉTRIQUES PAR CATÉGORIE")
print("="*70)

print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

report_dict = classification_report(
    y_test, y_pred, target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)

print(f"  {'Catégorie':<18} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8} {'Support':>9}")
print(f"  {'-'*68}")

per_class_metrics = {}
for cat in CLASS_NAMES:
    if cat not in report_dict:
        continue
    r      = report_dict[cat]
    cat_id = MAJOR_TO_ID[cat]
    mask_t = (y_test == cat_id)
    mask_p = (y_pred == cat_id)
    tp = ( mask_t &  mask_p).sum()
    fp = (~mask_t &  mask_p).sum()
    fn = ( mask_t & ~mask_p).sum()
    tn = (~mask_t & ~mask_p).sum()
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

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print(f"\n  Matrice de confusion :")
print(f"  {'':18}" + "".join(f"{n:>14}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:<18}" + "".join(f"{v:>14}" for v in row))

best_cat  = max(per_class_metrics, key=lambda k: per_class_metrics[k]['f1_score'])
worst_cat = min(per_class_metrics, key=lambda k: per_class_metrics[k]['f1_score'])
print(f"\n  ✅ Meilleure : {best_cat}  (F1={per_class_metrics[best_cat]['f1_score']:.4f})")
print(f"  ⚠️  Pire     : {worst_cat} (F1={per_class_metrics[worst_cat]['f1_score']:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 8. IMPORTANCE DES FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🔍 IMPORTANCE DES FEATURES")
print("="*70)

fi = pd.Series(
    final_model.feature_importances_,
    index=ALL_FEATURES
).sort_values(ascending=False)

for feat, imp in fi.items():
    bar = '█' * int(imp * 50)
    print(f"  {feat:25} : {imp:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════
# 9. COMPARAISON NB V4 vs DT V4
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("⚖️  COMPARAISON : NAIVE BAYES V4 vs ARBRE DE DÉCISION V4")
print("="*70)

nb_path = os.path.join(MODELS_DIR, 'model_info.json')
if os.path.exists(nb_path):
    with open(nb_path) as f:
        nb_info = json.load(f)
    nb_g  = nb_info.get('global_metrics', {})
    nb_pc = nb_info.get('per_class_metrics', {})

    print(f"\n  {'Métrique':<12} {'Naive Bayes':>14} {'Arbre Décision':>16}  Δ")
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
        w = '← DT' if d > 0 else ('← NB' if d < 0 else '=')
        print(f"  {label:<12} {nb_val:>14.4f} {dt_val:>16.4f}  {s}{d:.4f}  {w}")

    print(f"\n  Par catégorie :")
    print(f"  {'Catégorie':<18} {'NB F1':>8} {'DT F1':>8}  Δ      Meilleur")
    print(f"  {'-'*55}")
    for cat in CLASS_NAMES:
        dt_f1 = per_class_metrics[cat]['f1_score']
        nb_f1 = nb_pc.get(cat, {}).get('f1_score', 0)
        d = dt_f1 - nb_f1
        s = '+' if d >= 0 else ''
        w = 'DT ✅' if d > 0.005 else ('NB ✅' if d < -0.005 else 'Égal')
        print(f"  {cat:<18} {nb_f1:>8.4f} {dt_f1:>8.4f}  {s}{d:.4f}  {w}")
else:
    print(f"  Lance d'abord Naive_Bayes_v4.py pour avoir la comparaison NB vs DT")

# ══════════════════════════════════════════════════════════════════════════
# 10. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"💾 SAUVEGARDE → models/{VERSION}/")
print("="*70)

dt_model_path = os.path.join(MODELS_DIR, 'decision_tree_model.pkl')
with open(dt_model_path, 'wb') as f:
    pickle.dump(final_model, f)

dt_info = {
    'model_type':            'DecisionTreeClassifier',
    'version':               VERSION,
    'features':              ALL_FEATURES,
    'continuous_features':   CONTINUOUS_FEATURES,
    'binary_features':       BINARY_FEATURES,
    'best_normalization':    best_norm_name,
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
    'feature_importance': fi.to_dict(),
    'l1_l2_note': (
        'L1/L2 appliquées seulement sur les 3 features continues. '
        'Is_Hidden_Gem (binaire) réintégrée sans normalisation.'
    )
}

dt_info_path = os.path.join(MODELS_DIR, 'decision_tree_info.json')
with open(dt_info_path, 'w') as f:
    json.dump(dt_info, f, indent=2)

print(f"  ✅ models/{VERSION}/decision_tree_model.pkl")
print(f"  ✅ models/{VERSION}/decision_tree_info.json")

print(f"""
{"="*70}
✅ ARBRE DE DÉCISION V4
  Normalisation : {best_norm_name}
  Profondeur    : {final_model.get_depth()}
  Feuilles      : {final_model.get_n_leaves()}
  Accuracy      : {acc:.4f}  |  F1 : {f1:.4f}
  CV5           : {norm_results[best_norm_name]['cv5_mean']:.4f} ±{norm_results[best_norm_name]['cv5_std']:.4f}

  Features utilisées : {ALL_FEATURES}
  L1/L2 sur          : {CONTINUOUS_FEATURES} uniquement
{"="*70}

⏭️  Pour pousser sur Git :
    git add Decision_tree_v4.py
    git commit -m "feat: Decision Tree V4 adapté aux features optimisées V4"
    git push
""")