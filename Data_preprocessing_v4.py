"""
DATA PREPROCESSING — V4 FINAL
Dataset : imdb_top_1000_augmente_massif.csv
Sortie  : data/v4/

Corrections par rapport a V3 :
  - Suppression des 7 features correlees (corr > 0.85)
  - Normalisation corrigee : seulement les 3 features continues
  - Is_Hidden_Gem binaire 0/1 remplace Hidden_Gem_Score continu
  - Reequilibrage des classes par undersampling (Drama etait 43.8%)

Features finales pour le modele :
  - IMDB_Rating_MM  (normalisee)
  - Votes_Log_MM    (normalisee)
  - Movie_Age_MM    (normalisee)
  - Is_Hidden_Gem   (binaire 0/1, non normalisee)
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from pathlib import Path
import warnings, os

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# CONFIGURATION GENERALE
# ----------------------------------------------------------------------------

CSV_FILE   = 'imdb_top_1000_augmente_massif.csv'
VERSION    = 'v4'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data', VERSION)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

print(f"\n=== PREPROCESSING {VERSION.upper()} — Features optimisees + Normalisation corrigee ===\n")

# ----------------------------------------------------------------------------
# DEFINITION DES 4 CATEGORIES MAJEURES
# V4 etend la liste des sous-genres par rapport a V3
# (ajout de Music, Musical, Romance, Sport, War, History, Sci-Fi)
# ----------------------------------------------------------------------------

GENRE_GROUPS = {
    'Drama':        ['Drama', 'Biography', 'Mystery', 'Film-Noir', 'Family',
                     'Music', 'Musical', 'Romance', 'Sport', 'War', 'History'],
    'Action':       ['Action', 'Adventure', 'Western', 'Fantasy', 'Thriller', 'Sci-Fi'],
    'Comedy':       ['Comedy', 'Animation'],
    'Crime_Horror': ['Crime', 'Horror']
}

# Mapping inverse : sous-genre -> categorie majeure (ex: 'Horror' -> 'Crime_Horror')
SUBGENRE_TO_MAJOR = {sg: m for m, subs in GENRE_GROUPS.items() for sg in subs}

# Mapping categorie -> ID numerique (ex: 'Drama' -> 0)
MAJOR_GENRE_TO_ID = {genre: idx for idx, genre in enumerate(GENRE_GROUPS.keys())}
ID_TO_MAJOR_GENRE = {idx: genre for genre, idx in MAJOR_GENRE_TO_ID.items()}

print(f"{len(GENRE_GROUPS)} categories majeures :")
for major, subgenres in GENRE_GROUPS.items():
    print(f"  [{MAJOR_GENRE_TO_ID[major]}] {major:15} <- {', '.join(subgenres)}")

# ----------------------------------------------------------------------------
# 1. CHARGEMENT ET NETTOYAGE DES DONNEES
# ----------------------------------------------------------------------------

print("\n--- Chargement et nettoyage ---")

df = pd.read_csv(CSV_FILE)
initial_count = len(df)
print(f"Dataset brut      : {initial_count} films")

# Suppression des lignes sans genre (colonne critique pour la classification)
df_clean = df.dropna(subset=['Genre']).copy()

# Conversion des colonnes numeriques (les valeurs non convertibles deviennent NaN)
df_clean['IMDB_Rating']   = pd.to_numeric(df_clean['IMDB_Rating'],   errors='coerce')
df_clean['Released_Year'] = pd.to_numeric(df_clean['Released_Year'], errors='coerce')
df_clean['No_of_Votes']   = pd.to_numeric(df_clean['No_of_Votes'],   errors='coerce')
df_clean['Gross']         = pd.to_numeric(
    df_clean['Gross'].astype(str).str.replace(',', '').replace('', np.nan),
    errors='coerce'
)

# Suppression des lignes avec des valeurs manquantes sur les colonnes critiques
df_clean = df_clean.dropna(subset=['IMDB_Rating', 'Released_Year', 'No_of_Votes'])

# Suppression des doublons
df_clean = df_clean.drop_duplicates()

print(f"Apres nettoyage   : {len(df_clean)} films")
print(f"Lignes supprimees : {initial_count - len(df_clean)}")

# ----------------------------------------------------------------------------
# 2. TRAITEMENT DES GENRES
# ----------------------------------------------------------------------------

print("\n--- Traitement des genres ---")

# Extraction du genre principal (premier genre liste dans la colonne Genre)
df_clean['Genre_Primary'] = df_clean['Genre'].apply(
    lambda g: 'Unknown' if pd.isna(g) else str(g).split(',')[0].strip()
)

# Mapping vers la categorie majeure (Drama par defaut si inconnu)
df_clean['Genre_Major']    = df_clean['Genre_Primary'].apply(
    lambda g: SUBGENRE_TO_MAJOR.get(g, 'Drama')
)
df_clean['Genre_Major_ID'] = df_clean['Genre_Major'].map(MAJOR_GENRE_TO_ID)

# Affichage de la distribution avant reequilibrage
print("Distribution AVANT reequilibrage :")
dist_before = df_clean['Genre_Major'].value_counts()
for cat, cnt in dist_before.items():
    pct = cnt / len(df_clean) * 100
    print(f"  {cat:15} : {cnt:5} ({pct:.1f}%)")

# ----------------------------------------------------------------------------
# 3. REEQUILIBRAGE DES CLASSES PAR UNDERSAMPLING
# Probleme V3 : Drama = 43.8% -> biais massif vers Drama
# Solution V4 : on plafonne chaque classe a 1.5x la plus petite classe
# On garde en priorite les films les mieux notes
# ----------------------------------------------------------------------------

print("\n--- Reequilibrage des classes ---")

min_class   = dist_before.min()
max_allowed = int(min_class * 1.5)

print(f"Classe la plus petite : {min_class} films")
print(f"Plafond par classe    : {max_allowed} films")
print(f"Strategie             : conservation des films les mieux notes")

balanced_parts = []
for genre in GENRE_GROUPS.keys():
    subset = df_clean[df_clean['Genre_Major'] == genre]
    if len(subset) > max_allowed:
        subset = subset.nlargest(max_allowed, 'IMDB_Rating')
    balanced_parts.append(subset)

df_bal = pd.concat(balanced_parts).reset_index(drop=True)

print(f"\nDistribution APRES reequilibrage :")
dist_after = df_bal['Genre_Major'].value_counts()
for cat, cnt in dist_after.items():
    pct = cnt / len(df_bal) * 100
    print(f"  {cat:15} : {cnt:5} ({pct:.1f}%)")
print(f"\nTotal conserve : {len(df_bal)} films")

# ----------------------------------------------------------------------------
# 4. CONSTRUCTION DES 4 FEATURES FINALES
# Choix base sur l'analyse de correlation de la V3 :
#   - IMDB_Rating   : qualite pure, peu correlee avec Votes (0.32)
#   - Votes_Log     : log(votes) remplace No_of_Votes brut,
#                     casse les correlations lineaires extremes de V3
#   - Movie_Age     : dimension temporelle independante,
#                     Released_Year supprime (corr = -1.00 avec Movie_Age)
#   - Is_Hidden_Gem : binaire 0/1, remplace Hidden_Gem_Score continu
#                     (corr = 0.92 avec Rating en V3)
# ----------------------------------------------------------------------------

print("\n--- Construction des 4 features ---")

current_year = 2026

# Feature 1 : IMDB_Rating — qualite pure, sera normalisee
# (pas de calcul supplementaire, colonne existante)

# Feature 2 : log(1 + votes) — casse les correlations lineaires extremes
# Blockbuster_Score et Rating_x_Votes avaient corr=1.00 avec No_of_Votes en V3
df_bal['Votes_Log'] = np.log1p(df_bal['No_of_Votes'])

# Feature 3 : age du film — dimension temporelle independante
# Released_Year est supprime car corr = -1.00 avec Movie_Age (doublon parfait)
df_bal['Movie_Age'] = current_year - df_bal['Released_Year']

# Feature 4 : pepite cachee — binaire 0/1, non normalisee
# Remplace Hidden_Gem_Score continu (corr = 0.92 avec Rating en V3)
# Un film est une pepite si : bien note (>= 7.5) ET peu connu (votes < mediane)
votes_median = df_bal['No_of_Votes'].median()
df_bal['Is_Hidden_Gem'] = (
    (df_bal['IMDB_Rating'] >= 7.5) &
    (df_bal['No_of_Votes'] < votes_median)
).astype(int)

print("Features construites :")
print(f"  IMDB_Rating   -> qualite pure (sera normalisee)")
print(f"  Votes_Log     -> log(1 + votes), casse les correlations extremes")
print(f"  Movie_Age     -> {current_year} - Released_Year (sera normalisee)")
print(f"  Is_Hidden_Gem -> binaire 0/1 (non normalisee)")
print(f"\n  Mediane votes pour Is_Hidden_Gem : {int(votes_median):,}")
hg = df_bal['Is_Hidden_Gem'].value_counts()
print(f"  Films ordinaires (0) : {hg.get(0, 0)}")
print(f"  Pepites cachees  (1) : {hg.get(1, 0)}")

# ----------------------------------------------------------------------------
# 5. NORMALISATION MinMax — UNIQUEMENT LES 3 FEATURES CONTINUES
# Is_Hidden_Gem est deja en 0/1 donc pas besoin de normalisation
# Normaliser un binaire est inutile et peut causer des erreurs (division par 0)
# ----------------------------------------------------------------------------

print("\n--- Normalisation MinMax (3 features continues uniquement) ---")

COLS_TO_NORMALIZE = ['IMDB_Rating', 'Votes_Log', 'Movie_Age']
COLS_MM           = ['IMDB_Rating_MM', 'Votes_Log_MM', 'Movie_Age_MM']

scaler = MinMaxScaler()
df_bal[COLS_MM] = scaler.fit_transform(df_bal[COLS_TO_NORMALIZE])

for raw, mm in zip(COLS_TO_NORMALIZE, COLS_MM):
    print(f"  {raw:15} -> {mm}  [{df_bal[mm].min():.3f}, {df_bal[mm].max():.3f}]")

print(f"\n  Is_Hidden_Gem -> non normalisee (binaire 0/1)")

# Liste finale des features pour le modele
FEATURES_FOR_MODEL = ['IMDB_Rating_MM', 'Votes_Log_MM', 'Movie_Age_MM', 'Is_Hidden_Gem']
print(f"\n  Features pour le modele : {FEATURES_FOR_MODEL}")

# ----------------------------------------------------------------------------
# 6. VERIFICATIONS QUALITE
# Trois verifications pour s'assurer que les features sont utilisables :
#   a) Matrice de correlation (objectif : pas de paire > 0.7)
#   b) ANOVA F-score (objectif : chaque feature discrimine les genres)
#   c) Moyennes par genre (objectif : valeurs differentes entre genres)
# ----------------------------------------------------------------------------

print("\n--- Verifications qualite ---")

# a) Matrice de correlation entre les 4 features finales
print("\nMatrice de correlation (4 features finales) :")
corr = df_bal[FEATURES_FOR_MODEL].corr().round(3)
print(corr.to_string())

# Alerte si une paire depasse le seuil de 0.7
print("\nPaires correlees > 0.7 :")
corr_abs = corr.abs()
found = False
for i in range(len(FEATURES_FOR_MODEL)):
    for j in range(i+1, len(FEATURES_FOR_MODEL)):
        val = corr_abs.iloc[i, j]
        if val > 0.7:
            print(f"  ATTENTION : {FEATURES_FOR_MODEL[i]} <-> {FEATURES_FOR_MODEL[j]} : {val:.3f}")
            found = True
if not found:
    print("  Aucune paire > 0.7 — features independantes")

# b) ANOVA F-score : mesure la capacite de chaque feature a separer les genres
# Un F-score eleve et p < 0.05 signifie que la feature est utile pour la classification
print("\nANOVA F-score par feature :")
X_check = df_bal[FEATURES_FOR_MODEL].values
y_check = df_bal['Genre_Major_ID'].values
f_scores, p_values = f_classif(X_check, y_check)
for feat, f, p in sorted(
    zip(FEATURES_FOR_MODEL, f_scores, p_values), key=lambda x: -x[1]
):
    signif = "discriminant" if p < 0.05 else "non discriminant"
    print(f"  {feat:22} F={f:8.2f}  p={p:.4f}  {signif}")

# c) Moyennes par genre : les valeurs doivent differer entre genres
# Si toutes les moyennes sont identiques, la feature ne sert a rien
print("\nMoyennes par genre (doivent differer entre genres) :")
print(df_bal.groupby('Genre_Major')[FEATURES_FOR_MODEL].mean().round(3).to_string())

# ----------------------------------------------------------------------------
# 7. CONSTRUCTION DU DATASET FINAL ET SAUVEGARDE
# ----------------------------------------------------------------------------

print(f"\n--- Sauvegarde -> data/{VERSION}/ ---")

keep_cols = [
    'Series_Title', 'Released_Year', 'IMDB_Rating', 'No_of_Votes',
    'Director', 'Genre', 'Genre_Primary', 'Genre_Major', 'Genre_Major_ID',
    'Votes_Log', 'Movie_Age', 'Is_Hidden_Gem',
    *COLS_MM,
]

# Ajout des colonnes acteurs si elles existent dans le dataset
for star in ['Star1', 'Star2', 'Star3', 'Star4']:
    if star in df_bal.columns:
        keep_cols.append(star)

ml_dataset = df_bal[keep_cols].copy()

# Sauvegarde CSV et JSON du dataset principal
ml_dataset.to_csv(os.path.join(DATA_DIR, 'movies_processed_v4.csv'), index=False)
ml_dataset.to_json(
    os.path.join(DATA_DIR, 'movies_processed_v4.json'),
    orient='records', indent=2
)
print(f"  data/{VERSION}/movies_processed_v4.csv")
print(f"  data/{VERSION}/movies_processed_v4.json")

# Sauvegarde du mapping des genres
genre_mapping = {
    'major_to_id':       MAJOR_GENRE_TO_ID,
    'id_to_major':       {str(k): v for k, v in ID_TO_MAJOR_GENRE.items()},
    'subgenre_to_major': SUBGENRE_TO_MAJOR,
    'genre_groups':      GENRE_GROUPS,
}
with open(os.path.join(DATA_DIR, 'genre_mapping_v4.json'), 'w') as f:
    json.dump(genre_mapping, f, indent=2)
print(f"  data/{VERSION}/genre_mapping_v4.json")

# Sauvegarde des metadonnees du dataset
metadata = {
    'version':             VERSION,
    'total_movies':        len(ml_dataset),
    'features_for_model':  FEATURES_FOR_MODEL,
    'normalized_features': COLS_MM,
    'binary_features':     ['Is_Hidden_Gem'],
    'balancing':           f'undersampling -> max {max_allowed} films/classe',
    'votes_median':        float(votes_median),
    'class_distribution':  ml_dataset['Genre_Major'].value_counts().to_dict(),
    'removed_v3_features': [
        'Blockbuster_Score  -> corr=1.00 avec No_of_Votes',
        'Rating_x_Votes     -> corr=1.00 avec No_of_Votes',
        'Custom_Popularity  -> corr=0.91 avec IMDB_Rating',
        'Released_Year_MM   -> corr=-1.00 avec Movie_Age',
        'Hidden_Gem_Score   -> corr=0.92 avec IMDB_Rating',
        'Recent_Popularity  -> redondant',
        'High_Quality       -> redondant avec IMDB_Rating',
    ]
}
with open(os.path.join(DATA_DIR, 'metadata_v4.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# Sauvegarde d'un echantillon de 100 films pour tests rapides
ml_dataset.head(100).to_json(
    os.path.join(DATA_DIR, 'sample_100_movies_v4.json'),
    orient='records', indent=2
)
print(f"  data/{VERSION}/metadata_v4.json")
print(f"  data/{VERSION}/sample_100_movies_v4.json")

# Resume final
print(f"\n=== PREPROCESSING {VERSION.upper()} TERMINE ===")
print(f"  Films equilibres : {len(ml_dataset)}")
print(f"  Features modele  : {FEATURES_FOR_MODEL}")
print(f"  Nb features      : {len(FEATURES_FOR_MODEL)} (au lieu de 11 en V3)")
print(f"  Dossier          : data/{VERSION}/")
print(f"\n  Prochaine etape -> Naive_Bayes_v4.py")
print(f"    X = df{FEATURES_FOR_MODEL}")
print(f"    y = df['Genre_Major_ID']")