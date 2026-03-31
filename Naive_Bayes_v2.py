"""
NAIVE BAYES — V2
Dataset : 1249 films — dataset augmente (+250)
Prerequis : Data_preprocessing_v2.py doit avoir ete lance avant.
"""

import pandas as pd
import numpy as np
import json, pickle, os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------------------------------------------------------
# CONFIGURATION GENERALE
# ----------------------------------------------------------------------------

VERSION    = 'v2'
LABEL      = '1249 films — dataset augmente (+250)'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data',   VERSION)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', VERSION)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

print(f"\n=== NAIVE BAYES {VERSION.upper()} — {LABEL} ===\n")

# ----------------------------------------------------------------------------
# CHARGEMENT DES DONNEES ET DU MAPPING DES GENRES
# ----------------------------------------------------------------------------

csv_file = os.path.join(DATA_DIR, 'movies_processed_enriched.csv')
if not os.path.exists(csv_file):
    print(f"ERREUR : lancez d'abord Data_preprocessing_{VERSION}.py")
    exit(1)

df = pd.read_csv(csv_file)
print(f"{len(df)} films charges")

# Chargement du mapping genre -> ID (produit par le preprocessing)
with open(os.path.join(DATA_DIR, 'genre_mapping.json')) as f:
    genre_mapping = json.load(f)

CLASS_NAMES = list(genre_mapping['major_to_id'].keys())
MAJOR_TO_ID = genre_mapping['major_to_id']
ID_TO_MAJOR = {int(k): v for k, v in genre_mapping['id_to_major'].items()}

print(f"Classes : {CLASS_NAMES}")
print("\nDistribution :")
for cat, cnt in df['Genre_Major'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"  {cat:15} : {cnt:5} films ({pct:.1f}%)")

# ----------------------------------------------------------------------------
# DEFINITION DES FEATURES ET DE LA CIBLE
# ----------------------------------------------------------------------------

TARGET = 'Genre_Major_ID'
y = df[TARGET].values

# Features normalisees (MinMax) produites par le preprocessing
FEATURE_COLS = [
    'IMDB_Rating_MM', 'No_of_Votes_MM', 'Released_Year_MM',
    'Custom_Popularity_MM', 'Blockbuster_Score_MM', 'Hidden_Gem_Score_MM',
    'Votes_Log_MM', 'Rating_x_Votes_MM', 'Recent_Popularity_MM', 'High_Quality_MM'
]
print(f"\n{len(FEATURE_COLS)} features MinMax utilisees")

# ----------------------------------------------------------------------------
# CROSS-VALIDATION
# Evaluation de la stabilite du modele sur 5 et 10 decoupages differents
# StratifiedKFold garantit que chaque fold respecte la distribution des classes
# ----------------------------------------------------------------------------

print("\n--- Cross-validation (GaussianNB + MinMax) ---")

X = df[FEATURE_COLS].values
cv5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores_5  = cross_val_score(GaussianNB(), X, y, cv=cv5,  scoring='accuracy')
scores_10 = cross_val_score(GaussianNB(), X, y, cv=cv10, scoring='accuracy')

print(f"\n  5-fold  -> Accuracy: {scores_5.mean():.4f}  (+-{scores_5.std():.4f})")
print(f"            Scores  : {[round(s, 4) for s in scores_5]}")
print(f"\n  10-fold -> Accuracy: {scores_10.mean():.4f}  (+-{scores_10.std():.4f})")
print(f"            Scores  : {[round(s, 4) for s in scores_10]}")

# Sauvegarde des resultats de cross-validation pour le fichier model_info.json
cv_results = {
    'cv5_mean':    float(scores_5.mean()),
    'cv5_std':     float(scores_5.std()),
    'cv5_scores':  [round(float(s), 4) for s in scores_5],
    'cv10_mean':   float(scores_10.mean()),
    'cv10_std':    float(scores_10.std()),
    'cv10_scores': [round(float(s), 4) for s in scores_10]
}

# ----------------------------------------------------------------------------
# ENTRAINEMENT FINAL
# 80% des donnees pour l'entrainement, 20% pour le test
# stratify=y garantit que la proportion des classes est respectee dans les deux sets
# ----------------------------------------------------------------------------

print("\n--- Entrainement final (80% train / 20% test) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {len(X_train)} films")
print(f"  Test  : {len(X_test)} films")

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------------------------------------------------------
# METRIQUES GLOBALES
# Accuracy  : % de bonnes predictions sur l'ensemble du test
# Precision : quand le modele predit X, a-t-il raison ?
# Recall    : parmi tous les vrais X, combien le modele en a-t-il trouves ?
# F1        : moyenne harmonique precision/recall (metrique principale)
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
# Permet d'identifier quels genres sont bien ou mal predits
# TP : vrai positif | FP : faux positif | FN : faux negatif | TN : vrai negatif
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

# Matrice de confusion : lignes = vrais genres, colonnes = genres predits
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
# SAUVEGARDE DU MODELE ET DES METRIQUES
# ----------------------------------------------------------------------------

print(f"\n--- Sauvegarde -> models/{VERSION}/ ---")

# Sauvegarde du modele entraine au format binaire (.pkl)
model_path = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Sauvegarde des informations et metriques du modele au format JSON
model_info = {
    'model_type':    'GaussianNB',
    'version':       VERSION,
    'label':         LABEL,
    'normalization': 'MinMax',
    'features':      FEATURE_COLS,
    'num_classes':   len(CLASS_NAMES),
    'class_names':   CLASS_NAMES,
    'genre_mapping': genre_mapping,
    'cv_results':    cv_results,
    'global_metrics': {
        'accuracy':  float(acc),
        'precision': float(pre),
        'recall':    float(rec),
        'f1_score':  float(f1)
    },
    'per_class_metrics': per_class_metrics
}
info_path = os.path.join(MODELS_DIR, 'model_info.json')
with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"  models/{VERSION}/naive_bayes_model.pkl")
print(f"  models/{VERSION}/model_info.json")

print(f"\n=== NAIVE BAYES {VERSION.upper()} TERMINE ===")
print(f"  Accuracy : {acc:.4f}  |  F1 : {f1:.4f}")
print(f"  CV5      : {scores_5.mean():.4f} +-{scores_5.std():.4f}")

# ============================================================================
# API FLASK
# Expose le modele via des routes HTTP pour le frontend
# ============================================================================

print("\n--- Demarrage de l'API Flask ---")

app = Flask(__name__)
CORS(app)

# Chargement du modele et des donnees en memoire au demarrage de l'API
with open(model_path, 'rb') as f:
    TRAINED_MODEL = pickle.load(f)
with open(info_path) as f:
    MODEL_INFO = json.load(f)

df_api          = pd.read_csv(csv_file)
GENRE_MAPPING   = MODEL_INFO['genre_mapping']
CLASS_NAMES_API = MODEL_INFO['class_names']
print(f"{len(df_api)} films charges | Classes : {CLASS_NAMES_API}\n")


@app.route('/api/health', methods=['GET'])
def health():
    """Verifie que l'API est en ligne et retourne les infos de base."""
    return jsonify({
        'status':       'ok',
        'version':      VERSION,
        'label':        LABEL,
        'model':        'GaussianNB',
        'movies_in_db': len(df_api),
        'accuracy':     MODEL_INFO['global_metrics']['accuracy']
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Retourne toutes les metriques du modele (CV, globales, par classe)."""
    return jsonify({
        'cv_results':        MODEL_INFO['cv_results'],
        'global_metrics':    MODEL_INFO['global_metrics'],
        'per_class_metrics': MODEL_INFO['per_class_metrics']
    })


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Retourne les 10 meilleurs films selon les filtres du questionnaire.
    Body JSON attendu :
    {
        "min_rating"        : 7.0,
        "era"               : "modern" | "vintage" | "mixed",
        "niche"             : true | false,
        "favorite_director" : "Nolan",
        "genre_preference"  : "Action"
    }
    """
    try:
        data       = request.json
        min_rating = float(data.get('min_rating', 7.0))
        era        = data.get('era', 'mixed')
        niche      = data.get('niche', False)
        director   = data.get('favorite_director', '').lower()
        genre_pref = data.get('genre_preference', '').lower()

        # Filtre par note minimale
        filtered = df_api[df_api['IMDB_Rating'] >= min_rating].copy()

        # Filtre par ere (moderne ou vintage)
        if era == 'modern':
            filtered = filtered[filtered['Released_Year'] >= 2000]
        elif era == 'vintage':
            filtered = filtered[filtered['Released_Year'] < 2000]

        # Filtre niche (< mediane de votes) ou mainstream (>= mediane)
        med = df_api['No_of_Votes'].median()
        if niche:
            filtered = filtered[filtered['No_of_Votes'] < med]
        else:
            filtered = filtered[filtered['No_of_Votes'] >= med]

        # Filtre par genre
        if genre_pref:
            filtered = filtered[
                filtered['Genre_Major'].str.lower().str.contains(genre_pref, na=False) |
                filtered['Genre_Primary'].str.lower().str.contains(genre_pref, na=False)
            ]

        # Filtre par realisateur
        if director:
            filtered = filtered[
                filtered['Director'].str.lower().str.contains(director, na=False)
            ]

        # Top 10 des films les mieux notes
        top = filtered.nlargest(10, 'IMDB_Rating')
        recs = [
            {
                'title':       r['Series_Title'],
                'year':        int(r['Released_Year']),
                'rating':      float(r['IMDB_Rating']),
                'votes':       int(r['No_of_Votes']),
                'genre':       r['Genre_Primary'],
                'major_genre': r['Genre_Major'],
                'director':    r['Director']
            }
            for _, r in top.iterrows()
        ]
        return jsonify({'version': VERSION, 'total_results': len(recs), 'recommendations': recs})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Retourne la liste des genres disponibles (majeurs et sous-genres)."""
    return jsonify({
        'major_genres': CLASS_NAMES_API,
        'sub_genres':   sorted(GENRE_MAPPING['genre_to_id'].keys()),
        'genre_groups': GENRE_MAPPING['genre_groups']
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Retourne les statistiques globales du dataset et du modele."""
    return jsonify({
        'version':       VERSION,
        'total_movies':  len(df_api),
        'model_metrics': MODEL_INFO['global_metrics'],
        'per_class':     MODEL_INFO['per_class_metrics']
    })


if __name__ == '__main__':
    print(f"\nAPI {VERSION.upper()} -> http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)