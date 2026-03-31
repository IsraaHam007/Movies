"""
NAIVE BAYES — V3
Dataset : 6226 films — dataset massif
Prerequis : Data_preprocessing_v3.py doit avoir ete lance avant.
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

VERSION    = 'v3'
LABEL      = '6226 films — dataset massif'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data',   VERSION)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', VERSION)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

print(f"\n=== NAIVE BAYES {VERSION.upper()} — {LABEL} ===\n")

# ----------------------------------------------------------------------------
# CHARGEMENT DES DONNEES ET DU MAPPING DES GENRES
# On charge le CSV produit par Data_preprocessing_v3.py
# et le fichier JSON qui contient la correspondance genre <-> ID numerique
# Ex : {'Drama': 0, 'Action': 1, 'Comedy': 2, 'Crime_Horror': 3}
# ----------------------------------------------------------------------------

csv_file = os.path.join(DATA_DIR, 'movies_processed_enriched.csv')
if not os.path.exists(csv_file):
    print(f"ERREUR : lancez d'abord Data_preprocessing_{VERSION}.py")
    exit(1)

df = pd.read_csv(csv_file)
print(f"{len(df)} films charges")

with open(os.path.join(DATA_DIR, 'genre_mapping.json')) as f:
    genre_mapping = json.load(f)

# CLASS_NAMES : liste des noms de genres ['Drama', 'Action', 'Comedy', 'Crime_Horror']
# MAJOR_TO_ID : dict genre -> ID numerique, utilise pour calculer les metriques par classe
# ID_TO_MAJOR : dict ID -> genre, utilise pour decoder les predictions du modele
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
# X : matrice des features (ce que le modele utilise pour apprendre)
# y : vecteur cible (le genre a predire, sous forme d'ID numerique)
# Les features sont les colonnes normalisees (_MM) produites par le preprocessing
# ----------------------------------------------------------------------------

TARGET = 'Genre_Major_ID'
y = df[TARGET].values

FEATURE_COLS = [
    'IMDB_Rating_MM', 'No_of_Votes_MM', 'Released_Year_MM',
    'Custom_Popularity_MM', 'Blockbuster_Score_MM', 'Hidden_Gem_Score_MM',
    'Votes_Log_MM', 'Rating_x_Votes_MM', 'Recent_Popularity_MM', 'High_Quality_MM'
]
print(f"\n{len(FEATURE_COLS)} features MinMax utilisees")

# ----------------------------------------------------------------------------
# CROSS-VALIDATION
# La cross-validation evalue la robustesse du modele en le testant sur
# plusieurs decoupages differents des donnees.
#
# StratifiedKFold : chaque fold conserve la meme proportion de classes
# que le dataset complet (ex: si Drama = 43%, chaque fold aura ~43% de Drama)
# Cela evite qu'un fold soit compose uniquement d'un seul genre.
#
# 5-fold  : decoupe en 5 parties, entraine sur 4 et teste sur 1 (x5)
# 10-fold : meme principe mais avec 10 parties (plus precis, plus lent)
#
# La moyenne et l'ecart-type des scores indiquent si le modele est stable
# Un ecart-type faible signifie que le modele se comporte de maniere consistante
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

# Sauvegarde des resultats pour les inclure dans model_info.json
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
# On divise les donnees en deux ensembles :
#   - Train (80%) : le modele apprend les patterns sur ces donnees
#   - Test  (20%) : on evalue le modele sur des donnees qu'il n'a jamais vues
#
# stratify=y : s'assure que les deux sets ont la meme proportion de genres
# random_state=42 : fixe la graine aleatoire pour avoir des resultats reproductibles
#
# GaussianNB : modele Naive Bayes qui suppose que chaque feature suit
# une distribution gaussienne (courbe en cloche) au sein de chaque classe
# ----------------------------------------------------------------------------

print("\n--- Entrainement final (80% train / 20% test) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {len(X_train)} films")
print(f"  Test  : {len(X_test)} films")

model = GaussianNB()
model.fit(X_train, y_train)   # le modele apprend les distributions de chaque feature par genre
y_pred = model.predict(X_test) # le modele predit le genre pour chaque film du test

# ----------------------------------------------------------------------------
# METRIQUES GLOBALES
# Ces 4 metriques evaluent la qualite globale du modele sur le set de test :
#
# Accuracy  : (TP + TN) / total
#             % de films correctement classes parmi tous les films testes
#             Attention : peut etre trompeuse si les classes sont desequilibrees
#
# Precision : TP / (TP + FP)
#             Quand le modele dit "ce film est Drama", a-t-il raison ?
#             Une precision faible = beaucoup de faux positifs
#
# Recall    : TP / (TP + FN)
#             Parmi tous les vrais Drama, combien le modele en a-t-il trouves ?
#             Un recall faible = le modele rate beaucoup de vrais cas
#
# F1-Score  : 2 * (Precision * Recall) / (Precision + Recall)
#             Moyenne harmonique de Precision et Recall
#             C'est la metrique la plus importante pour ce projet
#             car elle penalise les modeles qui favorisent une seule classe
#
# average='weighted' : pondere chaque classe par son nombre de films (support)
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
# On calcule les metriques individuellement pour chaque genre afin
# d'identifier les classes bien predites et celles qui posent probleme
#
# Pour chaque genre, on construit une matrice de confusion binaire :
#   TP (Vrai Positif)  : film Drama predit Drama     -> bonne prediction
#   FP (Faux Positif)  : film non-Drama predit Drama -> fausse alarme
#   FN (Faux Negatif)  : film Drama predit non-Drama -> rate
#   TN (Vrai Negatif)  : film non-Drama predit non-Drama -> bonne prediction
#
# cat_acc = (TP + TN) / total : accuracy specifique a ce genre
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
    mask_true = (y_test == cat_id)   # films qui sont vraiment de ce genre
    mask_pred = (y_pred == cat_id)   # films que le modele predit comme ce genre
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

# Matrice de confusion complete : lignes = vrais genres, colonnes = genres predits
# Un bon modele a des valeurs elevees sur la diagonale (vrais positifs)
# et des valeurs proches de 0 en dehors (erreurs)
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
# Le modele est sauvegarde en .pkl (format binaire Python) pour pouvoir
# etre rechargé et utilise directement par l'API Flask sans re-entrainement
# model_info.json contient toutes les metriques et infos du modele
# pour que le frontend puisse les afficher
# ----------------------------------------------------------------------------

print(f"\n--- Sauvegarde -> models/{VERSION}/ ---")

model_path = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

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
# Flask est un micro-framework Python qui permet de creer une API REST.
# L'API expose le modele entraine via des routes HTTP que le frontend
# peut appeler avec des requetes GET ou POST.
#
# CORS (Cross-Origin Resource Sharing) : permet au frontend (qui tourne
# sur un port different, ex: localhost:3000) d'appeler l'API (localhost:5000)
# sans etre bloque par le navigateur pour des raisons de securite.
# ============================================================================

print("\n--- Demarrage de l'API Flask ---")

app = Flask(__name__)
CORS(app)

# Chargement du modele et des donnees une seule fois au demarrage
# pour ne pas les recharger a chaque requete (gain de performance)
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
    """
    Route de verification : retourne le statut de l'API et les infos du modele.
    Utile pour savoir si le serveur est bien demarre avant d'envoyer des requetes.
    """
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
    """
    Retourne toutes les metriques du modele.
    Utilise par le frontend pour afficher les performances du modele.
    """
    return jsonify({
        'cv_results':        MODEL_INFO['cv_results'],
        'global_metrics':    MODEL_INFO['global_metrics'],
        'per_class_metrics': MODEL_INFO['per_class_metrics']
    })


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Route principale : retourne les 10 meilleurs films selon les reponses
    du questionnaire de personnalite envoye par le frontend.

    Body JSON attendu :
    {
        "min_rating"        : 7.0,        note IMDB minimum
        "era"               : "modern",   "modern", "vintage" ou "mixed"
        "niche"             : true,       true = films peu connus, false = populaires
        "favorite_director" : "Nolan",    filtre optionnel par realisateur
        "genre_preference"  : "Action"    filtre optionnel par genre
    }
    """
    try:
        data       = request.json
        min_rating = float(data.get('min_rating', 7.0))
        era        = data.get('era', 'mixed')
        niche      = data.get('niche', False)
        director   = data.get('favorite_director', '').lower()
        genre_pref = data.get('genre_preference', '').lower()

        # Application des filtres successifs sur le dataset
        filtered = df_api[df_api['IMDB_Rating'] >= min_rating].copy()

        # Filtre temporel : films recents (apres 2000) ou anciens (avant 2000)
        if era == 'modern':
            filtered = filtered[filtered['Released_Year'] >= 2000]
        elif era == 'vintage':
            filtered = filtered[filtered['Released_Year'] < 2000]

        # Filtre popularite : niche = peu de votes, mainstream = beaucoup de votes
        med = df_api['No_of_Votes'].median()
        if niche:
            filtered = filtered[filtered['No_of_Votes'] < med]
        else:
            filtered = filtered[filtered['No_of_Votes'] >= med]

        # Filtre par genre (recherche dans le genre majeur et le sous-genre)
        if genre_pref:
            filtered = filtered[
                filtered['Genre_Major'].str.lower().str.contains(genre_pref, na=False) |
                filtered['Genre_Primary'].str.lower().str.contains(genre_pref, na=False)
            ]

        # Filtre par realisateur (recherche partielle, insensible a la casse)
        if director:
            filtered = filtered[
                filtered['Director'].str.lower().str.contains(director, na=False)
            ]

        # Retour des 10 films les mieux notes apres filtrage
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
    """
    Retourne la liste complete des genres disponibles.
    Utilise par le frontend pour peupler les listes deroulantes du questionnaire.
    """
    return jsonify({
        'major_genres': CLASS_NAMES_API,
        'sub_genres':   sorted(GENRE_MAPPING['genre_to_id'].keys()),
        'genre_groups': GENRE_MAPPING['genre_groups']
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Retourne les statistiques generales du dataset et les performances du modele.
    """
    return jsonify({
        'version':       VERSION,
        'total_movies':  len(df_api),
        'model_metrics': MODEL_INFO['global_metrics'],
        'per_class':     MODEL_INFO['per_class_metrics']
    })


if __name__ == '__main__':
    print(f"\nAPI {VERSION.upper()} -> http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)