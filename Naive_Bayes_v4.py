"""
NAIVE BAYES — V4
Dataset : classes equilibrees — features optimisees
Prerequis : Data_preprocessing_v4.py doit avoir ete lance avant.

Differences avec V3 :
  - 4 features independantes au lieu de 10 correlees
  - Fichiers V4 : movies_processed_v4.csv + genre_mapping_v4.json
  - Classes equilibrees par undersampling (Drama n'est plus a 43.8%)
  - Nouvelle route /api/predict pour predire le genre d'un film a la volee
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

VERSION    = 'v4'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data',   VERSION)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', VERSION)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# Les 4 features selectionnees apres analyse de correlation en V3
# IMDB_Rating_MM, Votes_Log_MM, Movie_Age_MM : normalisees entre 0 et 1
# Is_Hidden_Gem : binaire 0/1, non normalisee (deja dans la bonne echelle)
FEATURE_COLS = [
    'IMDB_Rating_MM',  # qualite pure du film normalisee
    'Votes_Log_MM',    # popularite sans outliers normalisee
    'Movie_Age_MM',    # age du film normalisee
    'Is_Hidden_Gem'    # 1 si bien note ET peu connu, sinon 0
]

print(f"\n=== NAIVE BAYES {VERSION.upper()} — Features optimisees + Classes equilibrees ===\n")

# ----------------------------------------------------------------------------
# CHARGEMENT DES DONNEES V4
# On charge les fichiers produits par Data_preprocessing_v4.py
# On verifie aussi que toutes les features attendues existent dans le CSV
# ----------------------------------------------------------------------------

csv_file     = os.path.join(DATA_DIR, 'movies_processed_v4.csv')
mapping_file = os.path.join(DATA_DIR, 'genre_mapping_v4.json')

if not os.path.exists(csv_file):
    print(f"ERREUR : fichier introuvable -> {csv_file}")
    print(f"Lance d'abord : python Data_preprocessing_v4.py")
    exit(1)

df = pd.read_csv(csv_file)
print(f"{len(df)} films charges depuis movies_processed_v4.csv")

# Chargement du mapping genre <-> ID numerique
with open(mapping_file) as f:
    genre_mapping = json.load(f)

MAJOR_TO_ID = genre_mapping['major_to_id']
ID_TO_MAJOR = {int(k): v for k, v in genre_mapping['id_to_major'].items()}
CLASS_NAMES = list(MAJOR_TO_ID.keys())

print(f"Classes : {CLASS_NAMES}")
print("\nDistribution (apres equilibrage V4) :")
for cat, cnt in df['Genre_Major'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"  {cat:15} : {cnt:5} ({pct:.1f}%)")

# Verification que toutes les features attendues sont bien dans le CSV
# Si une feature manque, c'est que le preprocessing n'a pas ete relance
missing = [f for f in FEATURE_COLS if f not in df.columns]
if missing:
    print(f"\nERREUR : features manquantes dans le CSV : {missing}")
    print(f"Relance Data_preprocessing_v4.py")
    exit(1)

print(f"\nFeatures utilisees ({len(FEATURE_COLS)}) : {FEATURE_COLS}")

# ----------------------------------------------------------------------------
# PREPARATION X / y
# X : matrice des 4 features pour chaque film (ce que le modele utilise)
# y : vecteur des genres cibles sous forme d'ID numerique
# Ex : Drama=0, Action=1, Comedy=2, Crime_Horror=3
# ----------------------------------------------------------------------------

X = df[FEATURE_COLS].values
y = df['Genre_Major_ID'].values

print(f"\nX shape : {X.shape}  ({X.shape[0]} films, {X.shape[1]} features)")
print(f"y shape : {y.shape}  ({y.shape[0]} genres cibles)")

# ----------------------------------------------------------------------------
# CROSS-VALIDATION
# Meme principe que V3 : on evalue la stabilite du modele sur plusieurs decoupages
# L'amelioration attendue vient du fait que les classes sont maintenant equilibrees
# et les features sont independantes -> le modele generalise mieux
# ----------------------------------------------------------------------------

print("\n--- Cross-validation (GaussianNB) ---")

cv5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores_5  = cross_val_score(GaussianNB(), X, y, cv=cv5,  scoring='accuracy')
scores_10 = cross_val_score(GaussianNB(), X, y, cv=cv10, scoring='accuracy')

print(f"\n  5-fold  -> Accuracy: {scores_5.mean():.4f}  (+-{scores_5.std():.4f})")
print(f"            Scores  : {[round(s, 4) for s in scores_5]}")
print(f"\n  10-fold -> Accuracy: {scores_10.mean():.4f}  (+-{scores_10.std():.4f})")
print(f"            Scores  : {[round(s, 4) for s in scores_10]}")

cv_results = {
    'cv5_mean':    float(scores_5.mean()),
    'cv5_std':     float(scores_5.std()),
    'cv5_scores':  [round(float(s), 4) for s in scores_5],
    'cv10_mean':   float(scores_10.mean()),
    'cv10_std':    float(scores_10.std()),
    'cv10_scores': [round(float(s), 4) for s in scores_10],
}

# ----------------------------------------------------------------------------
# ENTRAINEMENT FINAL
# Meme logique que V3 : 80% train, 20% test, stratify pour respecter
# la distribution des classes dans les deux sets
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
# On compare automatiquement avec les resultats de V3 pour voir l'amelioration
# V3 : Accuracy=41.97%, F1=0.3147
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

# Comparaison directe V3 -> V4 pour mesurer l'impact des optimisations
print(f"\n  Comparaison V3 -> V4 :")
print(f"    Accuracy V3 : 41.97%  ->  V4 : {acc*100:.2f}%  {'amelioration' if acc > 0.4197 else 'similaire'}")
print(f"    F1      V3 : 0.3147   ->  V4 : {f1:.4f}  {'amelioration' if f1 > 0.3147 else 'similaire'}")

# ----------------------------------------------------------------------------
# METRIQUES PAR CATEGORIE
# En V4, on s'attend a ce que les 4 genres soient mieux equilibres
# car l'undersampling a reduit le biais vers Drama
# Un bon modele a des F1 proches entre les 4 genres (pas 0.58 vs 0.03 comme en V3)
# ----------------------------------------------------------------------------

print("\n--- Metriques par categorie ---")

print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

report_dict = classification_report(
    y_test, y_pred, target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)

print(f"  {'Categorie':<18} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8} {'Support':>9}")
print(f"  {'-'*68}")

per_class_metrics = {}
for cat in CLASS_NAMES:
    if cat not in report_dict:
        continue
    r      = report_dict[cat]
    cat_id = MAJOR_TO_ID[cat]
    mask_t = (y_test == cat_id)  # films qui sont vraiment de ce genre
    mask_p = (y_pred == cat_id)  # films que le modele predit comme ce genre
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

# Matrice de confusion : lignes = vrais genres, colonnes = genres predits
# La diagonale = bonnes predictions, hors diagonale = erreurs
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
# SAUVEGARDE DU MODELE
# vs_v3 : comparaison sauvegardee dans le JSON pour que le frontend
# puisse afficher l'evolution entre les versions
# ----------------------------------------------------------------------------

print(f"\n--- Sauvegarde -> models/{VERSION}/ ---")

model_path = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

model_info = {
    'model_type':          'GaussianNB',
    'version':             VERSION,
    'features':            FEATURE_COLS,
    'normalized_features': ['IMDB_Rating_MM', 'Votes_Log_MM', 'Movie_Age_MM'],
    'binary_features':     ['Is_Hidden_Gem'],
    'num_classes':         len(CLASS_NAMES),
    'class_names':         CLASS_NAMES,
    'genre_mapping':       genre_mapping,
    'cv_results':          cv_results,
    'global_metrics': {
        'accuracy':  float(acc),
        'precision': float(pre),
        'recall':    float(rec),
        'f1_score':  float(f1)
    },
    'per_class_metrics': per_class_metrics,
    # Comparaison V3 vs V4 sauvegardee pour affichage dans le frontend
    'vs_v3': {
        'accuracy_v3': 0.4197,
        'accuracy_v4': float(acc),
        'f1_v3':       0.3147,
        'f1_v4':       float(f1),
    }
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
# Meme principe que V3 avec deux nouveautes :
#   - Route /api/predict : predit le genre d'un film donne par le frontend
#   - Route /api/recommendations : inclut le filtre par saison
#   - Retourne aussi predicted_genre pour chaque film recommande
# ============================================================================

print("\n--- Demarrage de l'API Flask ---")

app = Flask(__name__)
CORS(app)

# Chargement en memoire une seule fois au demarrage
with open(model_path, 'rb') as f:
    TRAINED_MODEL = pickle.load(f)
with open(info_path) as f:
    MODEL_INFO = json.load(f)

df_api          = pd.read_csv(csv_file)
CLASS_NAMES_API = MODEL_INFO['class_names']

# Mediane des votes calculee une seule fois pour la regle Is_Hidden_Gem
# Doit etre la meme valeur que celle utilisee dans le preprocessing
votes_median = df_api['No_of_Votes'].median()

print(f"{len(df_api)} films charges | Classes : {CLASS_NAMES_API}\n")


def build_feature_vector(imdb_rating, no_of_votes, released_year):
    """
    Reconstruit le vecteur de 4 features pour un film donne.
    Reproduit exactement la logique de Data_preprocessing_v4.py
    pour que les valeurs soient dans la meme echelle que les donnees d'entrainement.

    Parametres :
        imdb_rating   : note IMDB du film (ex: 8.5)
        no_of_votes   : nombre de votes IMDB (ex: 150000)
        released_year : annee de sortie (ex: 2010)

    Retourne :
        liste de 4 valeurs [rating_mm, votes_mm, age_mm, is_hidden_gem]
    """
    # Recuperation des min/max du dataset pour reproduire la normalisation MinMax
    rating_min = float(df_api['IMDB_Rating'].min())
    rating_max = float(df_api['IMDB_Rating'].max())

    votes_log     = np.log1p(no_of_votes)
    votes_log_min = np.log1p(df_api['No_of_Votes'].min())
    votes_log_max = np.log1p(df_api['No_of_Votes'].max())

    movie_age = 2026 - released_year
    age_min   = float(df_api['Movie_Age'].min())
    age_max   = float(df_api['Movie_Age'].max())

    # Application de la formule MinMax : (valeur - min) / (max - min)
    rating_mm = (imdb_rating - rating_min) / (rating_max - rating_min)
    votes_mm  = (votes_log   - votes_log_min) / (votes_log_max - votes_log_min)
    age_mm    = (movie_age   - age_min) / (age_max - age_min)

    # Is_Hidden_Gem : meme regle que dans le preprocessing
    # 1 si bien note (>= 7.5) ET peu connu (votes < mediane du dataset)
    is_hidden_gem = int(imdb_rating >= 7.5 and no_of_votes < votes_median)

    return [rating_mm, votes_mm, age_mm, is_hidden_gem]


@app.route('/api/health', methods=['GET'])
def health():
    """Verifie que l'API est en ligne. Retourne les infos du modele."""
    return jsonify({
        'status':       'ok',
        'version':      VERSION,
        'model':        'GaussianNB',
        'features':     FEATURE_COLS,
        'movies_in_db': len(df_api),
        'accuracy':     MODEL_INFO['global_metrics']['accuracy']
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Retourne toutes les metriques du modele, incluant la comparaison V3 vs V4."""
    return jsonify({
        'cv_results':        MODEL_INFO['cv_results'],
        'global_metrics':    MODEL_INFO['global_metrics'],
        'per_class_metrics': MODEL_INFO['per_class_metrics'],
        'vs_v3':             MODEL_INFO['vs_v3']
    })


@app.route('/api/predict', methods=['POST'])
def predict_genre():
    """
    Nouvelle route V4 : predit le genre d'un film a partir de ses caracteristiques.
    Le frontend peut l'utiliser pour afficher le genre predit dans l'interface.

    Body JSON attendu :
    {
        "imdb_rating"   : 8.5,
        "no_of_votes"   : 50000,
        "released_year" : 2010
    }

    Retourne le genre predit + les probabilites pour chaque genre.
    predict_proba donne la confiance du modele ex: Drama=60%, Action=20%...
    """
    try:
        data          = request.json
        imdb_rating   = float(data.get('imdb_rating', 7.5))
        no_of_votes   = int(data.get('no_of_votes', 10000))
        released_year = int(data.get('released_year', 2000))

        # Construction du vecteur de features et prediction
        features   = build_feature_vector(imdb_rating, no_of_votes, released_year)
        prediction = TRAINED_MODEL.predict([features])[0]
        proba      = TRAINED_MODEL.predict_proba([features])[0]

        return jsonify({
            'predicted_genre': ID_TO_MAJOR[int(prediction)],
            'genre_id':        int(prediction),
            'probabilities': {
                CLASS_NAMES_API[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            },
            # Features retournees pour le debug et l'affichage dans le frontend
            'features_used': {
                'IMDB_Rating_MM': round(features[0], 4),
                'Votes_Log_MM':   round(features[1], 4),
                'Movie_Age_MM':   round(features[2], 4),
                'Is_Hidden_Gem':  features[3]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Route principale du questionnaire : filtre les films selon les preferences
    de l'utilisateur et retourne les 10 meilleurs films correspondants.

    Nouveaute V4 : filtre par saison/ambiance et predicted_genre dans la reponse.

    Body JSON attendu :
    {
        "min_rating"        : 7.5,
        "era"               : "modern" | "vintage" | "mixed",
        "niche"             : true | false,
        "favorite_director" : "Nolan",
        "genre_preference"  : "Action",
        "season"            : "halloween" | "christmas" | "summer" | "any"
    }
    """
    try:
        data       = request.json
        min_rating = float(data.get('min_rating', 7.0))
        era        = data.get('era', 'mixed').lower()
        niche      = data.get('niche', False)
        director   = data.get('favorite_director', '').lower().strip()
        genre_pref = data.get('genre_preference', '').lower().strip()
        season     = data.get('season', 'any').lower()

        # Application des filtres successifs
        filtered = df_api[df_api['IMDB_Rating'] >= min_rating].copy()

        # Filtre temporel
        if era == 'modern':
            filtered = filtered[filtered['Released_Year'] >= 2000]
        elif era == 'vintage':
            filtered = filtered[filtered['Released_Year'] < 2000]

        # Filtre popularite : niche ou mainstream
        if niche:
            filtered = filtered[filtered['No_of_Votes'] < votes_median]
        else:
            filtered = filtered[filtered['No_of_Votes'] >= votes_median]

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

        # Filtre par saison : chaque saison correspond a des genres specifiques
        # Halloween -> horreur/crime, Christmas -> comedie/drame, Summer -> action
        SEASON_GENRES = {
            'halloween': ['Crime_Horror'],
            'christmas': ['Comedy', 'Drama'],
            'summer':    ['Action'],
            'any':       CLASS_NAMES_API
        }
        allowed  = SEASON_GENRES.get(season, CLASS_NAMES_API)
        filtered = filtered[filtered['Genre_Major'].isin(allowed)]

        # Top 10 des films les mieux notes apres tous les filtres
        top  = filtered.nlargest(10, 'IMDB_Rating')
        recs = []
        for _, r in top.iterrows():
            # Nouveaute V4 : le modele predit aussi le genre de chaque film recommande
            # Permet de comparer le genre reel et le genre predit dans le frontend
            feats      = build_feature_vector(
                float(r['IMDB_Rating']),
                int(r['No_of_Votes']),
                int(r['Released_Year'])
            )
            pred_genre = ID_TO_MAJOR[int(TRAINED_MODEL.predict([feats])[0])]

            recs.append({
                'title':           r['Series_Title'],
                'year':            int(r['Released_Year']),
                'rating':          float(r['IMDB_Rating']),
                'votes':           int(r['No_of_Votes']),
                'genre':           r['Genre_Primary'],
                'major_genre':     r['Genre_Major'],
                'predicted_genre': pred_genre,
                'director':        r['Director'],
                'is_hidden_gem':   int(r['Is_Hidden_Gem']),
                'movie_age':       int(r['Movie_Age'])
            })

        return jsonify({
            'version':         VERSION,
            'total_results':   len(recs),
            'filters_applied': {
                'min_rating': min_rating, 'era': era,
                'niche':      niche,      'season': season,
                'genre_pref': genre_pref, 'director': director
            },
            'recommendations': recs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Retourne la liste des genres et leur mapping (pour le questionnaire frontend)."""
    return jsonify({
        'major_genres':      CLASS_NAMES_API,
        'genre_groups':      genre_mapping['genre_groups'],
        'subgenre_to_major': genre_mapping['subgenre_to_major']
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Retourne les statistiques generales et la comparaison V3 vs V4."""
    return jsonify({
        'version':       VERSION,
        'total_movies':  len(df_api),
        'features':      FEATURE_COLS,
        'model_metrics': MODEL_INFO['global_metrics'],
        'per_class':     MODEL_INFO['per_class_metrics'],
        'vs_v3':         MODEL_INFO['vs_v3']
    })


if __name__ == '__main__':
    print(f"\nAPI {VERSION.upper()} -> http://localhost:5000")
    print(f"Routes disponibles :")
    print(f"  GET  /api/health")
    print(f"  GET  /api/metrics")
    print(f"  GET  /api/genres")
    print(f"  GET  /api/stats")
    print(f"  POST /api/predict          {{ imdb_rating, no_of_votes, released_year }}")
    print(f"  POST /api/recommendations  {{ min_rating, era, niche, genre_preference,")
    print(f"                               favorite_director, season }}\n")
    app.run(debug=True, host='0.0.0.0', port=5000)