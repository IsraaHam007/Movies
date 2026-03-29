"""
🎬 DATA PREPROCESSING — V4 FINAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRECTIONS PAR RAPPORT À V3 :

  FEATURES :
  ❌ Supprimé : Blockbuster_Score    (corr = 1.00 avec No_of_Votes)
  ❌ Supprimé : Rating_x_Votes       (corr = 1.00 avec No_of_Votes)
  ❌ Supprimé : Custom_Popularity    (corr = 0.91 avec IMDB_Rating)
  ❌ Supprimé : Released_Year_MM     (corr = -1.00 avec Movie_Age)
  ❌ Supprimé : Hidden_Gem_Score     (corr = 0.92 avec IMDB_Rating)
  ❌ Supprimé : Recent_Popularity    (redondant)
  ❌ Supprimé : High_Quality         (redondant avec IMDB_Rating)

  NORMALISATION :
  ❌ V3 : normalisait 11 colonnes dont des doublons et des binaires
  ✅ V4 : normalise seulement les 3 features continues (Rating, Votes, Age)
  ✅ V4 : Is_Hidden_Gem est binaire 0/1 → pas de normalisation

  CLASSES :
  ❌ V3 : Drama = 43.8% → biais massif
  ✅ V4 : undersampling → classes équilibrées

  FEATURES FINALES POUR LE MODÈLE :
  → IMDB_Rating_MM   (normalisée)
  → Votes_Log_MM     (normalisée)
  → Movie_Age_MM     (normalisée)
  → Is_Hidden_Gem    (binaire, PAS normalisée)

CSV source : imdb_top_1000_augmente_massif.csv
Sortie     : data/v4/
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from pathlib import Path
import warnings, os
warnings.filterwarnings('ignore')

CSV_FILE   = 'imdb_top_1000_augmente_massif.csv'
VERSION    = 'v4'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data', VERSION)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

print("\n" + "🎬"*40)
print("  PREPROCESSING V4 — Features optimisées + Normalisation corrigée")
print("🎬"*40 + "\n")

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION DES 4 CATÉGORIES
# ══════════════════════════════════════════════════════════════════════════

GENRE_GROUPS = {
    'Drama':        ['Drama','Biography','Mystery','Film-Noir','Family',
                     'Music','Musical','Romance','Sport','War','History'],
    'Action':       ['Action','Adventure','Western','Fantasy','Thriller','Sci-Fi'],
    'Comedy':       ['Comedy','Animation'],
    'Crime_Horror': ['Crime','Horror']
}
SUBGENRE_TO_MAJOR = {sg: m for m, subs in GENRE_GROUPS.items() for sg in subs}
MAJOR_GENRE_TO_ID = {genre: idx for idx, genre in enumerate(GENRE_GROUPS.keys())}
ID_TO_MAJOR_GENRE = {idx: genre for genre, idx in MAJOR_GENRE_TO_ID.items()}

print("📌 4 catégories majeures :")
for major, subgenres in GENRE_GROUPS.items():
    print(f"   [{MAJOR_GENRE_TO_ID[major]}] {major:15} ← {', '.join(subgenres)}")

# ══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & NETTOYAGE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📥 CHARGEMENT & NETTOYAGE")
print("="*70)

df = pd.read_csv(CSV_FILE)
initial_count = len(df)
print(f"  Dataset brut      : {initial_count} films")

df_clean = df.dropna(subset=['Genre']).copy()
df_clean['IMDB_Rating']   = pd.to_numeric(df_clean['IMDB_Rating'],   errors='coerce')
df_clean['Released_Year'] = pd.to_numeric(df_clean['Released_Year'], errors='coerce')
df_clean['No_of_Votes']   = pd.to_numeric(df_clean['No_of_Votes'],   errors='coerce')
df_clean['Gross']         = pd.to_numeric(
    df_clean['Gross'].astype(str).str.replace(',', '').replace('', np.nan),
    errors='coerce'
)
df_clean = df_clean.dropna(subset=['IMDB_Rating', 'Released_Year', 'No_of_Votes'])
df_clean = df_clean.drop_duplicates()

print(f"  Après nettoyage   : {len(df_clean)} films")
print(f"  Lignes supprimées : {initial_count - len(df_clean)}")

# ══════════════════════════════════════════════════════════════════════════
# 2. TRAITEMENT DES GENRES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🎬 TRAITEMENT DES GENRES")
print("="*70)

df_clean['Genre_Primary'] = df_clean['Genre'].apply(
    lambda g: 'Unknown' if pd.isna(g) else str(g).split(',')[0].strip()
)
df_clean['Genre_Major'] = df_clean['Genre_Primary'].apply(
    lambda g: SUBGENRE_TO_MAJOR.get(g, 'Drama')
)
df_clean['Genre_Major_ID'] = df_clean['Genre_Major'].map(MAJOR_GENRE_TO_ID)

print("\n  Distribution AVANT rééquilibrage :")
dist_before = df_clean['Genre_Major'].value_counts()
for cat, cnt in dist_before.items():
    pct = cnt / len(df_clean) * 100
    bar = '█' * (cnt // 100)
    print(f"    {cat:15} : {cnt:5} ({pct:.1f}%)  {bar}")

# ══════════════════════════════════════════════════════════════════════════
# 3. RÉÉQUILIBRAGE DES CLASSES (undersampling)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("⚖️  RÉÉQUILIBRAGE DES CLASSES")
print("="*70)

min_class   = dist_before.min()
max_allowed = int(min_class * 1.5)

print(f"  Classe la plus petite : {min_class} films")
print(f"  Plafond par classe    : {max_allowed} films")
print(f"  Stratégie             : on garde les mieux notés en priorité")

balanced_parts = []
for genre in GENRE_GROUPS.keys():
    subset = df_clean[df_clean['Genre_Major'] == genre]
    if len(subset) > max_allowed:
        subset = subset.nlargest(max_allowed, 'IMDB_Rating')
    balanced_parts.append(subset)

df_bal = pd.concat(balanced_parts).reset_index(drop=True)

print(f"\n  Distribution APRÈS rééquilibrage :")
dist_after = df_bal['Genre_Major'].value_counts()
for cat, cnt in dist_after.items():
    pct = cnt / len(df_bal) * 100
    bar = '█' * (cnt // 30)
    print(f"    {cat:15} : {cnt:5} ({pct:.1f}%)  {bar}")
print(f"\n  Total conservé : {len(df_bal)} films")

# ══════════════════════════════════════════════════════════════════════════
# 4. CONSTRUCTION DES 4 FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🔬 CONSTRUCTION DES 4 FEATURES")
print("="*70)

current_year = 2026

# Feature 1 : IMDB_Rating — qualité pure, sera normalisée
# Corrélation avec No_of_Votes = 0.32 → acceptable pour Naive Bayes

# Feature 2 : Votes_Log = log(1 + votes)
# Remplace No_of_Votes brut → le log casse les corrélations linéaires
# extrêmes de V3 (Blockbuster=1.00, Rating_x_Votes=1.00)
df_bal['Votes_Log'] = np.log1p(df_bal['No_of_Votes'])

# Feature 3 : Movie_Age = 2026 - Released_Year
# On garde Movie_Age et supprime Released_Year
# (corr = -1.00 en V3 → doublon parfait)
df_bal['Movie_Age'] = current_year - df_bal['Released_Year']

# Feature 4 : Is_Hidden_Gem — BINAIRE, pas de normalisation
# Remplace Hidden_Gem_Score continu (corr = 0.92 avec Rating en V3)
# Définition : bien noté (>= 7.5) ET peu connu (votes < médiane)
votes_median = df_bal['No_of_Votes'].median()
df_bal['Is_Hidden_Gem'] = (
    (df_bal['IMDB_Rating'] >= 7.5) &
    (df_bal['No_of_Votes'] < votes_median)
).astype(int)

print("  Features construites :")
print(f"    ✓ IMDB_Rating    → qualité pure (sera normalisée)")
print(f"    ✓ Votes_Log      → log(1 + votes), casse les corrélations extrêmes")
print(f"    ✓ Movie_Age      → {current_year} - Released_Year (sera normalisée)")
print(f"    ✓ Is_Hidden_Gem  → binaire 0/1 (PAS normalisée)")
print(f"\n    Médiane votes utilisée pour Is_Hidden_Gem : {int(votes_median):,}")
hg = df_bal['Is_Hidden_Gem'].value_counts()
print(f"    Films ordinaires (0) : {hg.get(0, 0)}")
print(f"    Pépites cachées  (1) : {hg.get(1, 0)}")

# ══════════════════════════════════════════════════════════════════════════
# 5. NORMALISATION — SEULEMENT LES 3 FEATURES CONTINUES
#    Is_Hidden_Gem est déjà 0/1 → inutile et risqué de normaliser
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📐 NORMALISATION MinMax (3 features continues uniquement)")
print("="*70)

COLS_TO_NORMALIZE = ['IMDB_Rating', 'Votes_Log', 'Movie_Age']
COLS_MM           = ['IMDB_Rating_MM', 'Votes_Log_MM', 'Movie_Age_MM']

scaler = MinMaxScaler()
df_bal[COLS_MM] = scaler.fit_transform(df_bal[COLS_TO_NORMALIZE])

for raw, mm in zip(COLS_TO_NORMALIZE, COLS_MM):
    print(f"  {raw:15} → {mm}  [{df_bal[mm].min():.3f}, {df_bal[mm].max():.3f}]")

print(f"\n  Is_Hidden_Gem → NON normalisée (binaire 0/1)")

# Features finales pour le modèle
FEATURES_FOR_MODEL = ['IMDB_Rating_MM', 'Votes_Log_MM', 'Movie_Age_MM', 'Is_Hidden_Gem']
print(f"\n  ✅ Features pour Naive Bayes : {FEATURES_FOR_MODEL}")

# ══════════════════════════════════════════════════════════════════════════
# 6. VÉRIFICATIONS QUALITÉ
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("✅ VÉRIFICATIONS QUALITÉ")
print("="*70)

# 6a. Corrélation entre les 4 features finales
print("\n  Matrice de corrélation (4 features finales) :")
corr = df_bal[FEATURES_FOR_MODEL].corr().round(3)
print(corr.to_string())

print("\n  Paires corrélées > 0.7 :")
corr_abs = corr.abs()
found = False
for i in range(len(FEATURES_FOR_MODEL)):
    for j in range(i+1, len(FEATURES_FOR_MODEL)):
        val = corr_abs.iloc[i, j]
        if val > 0.7:
            print(f"    ⚠️  {FEATURES_FOR_MODEL[i]} ↔ {FEATURES_FOR_MODEL[j]} : {val:.3f}")
            found = True
if not found:
    print("    ✅ Aucune paire > 0.7 — features bien indépendantes !")

# 6b. ANOVA F-score
print("\n  ANOVA F-score par feature :")
X_check = df_bal[FEATURES_FOR_MODEL].values
y_check = df_bal['Genre_Major_ID'].values
f_scores, p_values = f_classif(X_check, y_check)
for feat, f, p in sorted(
    zip(FEATURES_FOR_MODEL, f_scores, p_values), key=lambda x: -x[1]
):
    signif = "✅ discriminant" if p < 0.05 else "⚠️  non discriminant"
    print(f"    {feat:22} F={f:8.2f}  p={p:.4f}  {signif}")

# 6c. Moyenne par genre
print("\n  Moyennes par genre (doivent différer entre genres) :")
print(df_bal.groupby('Genre_Major')[FEATURES_FOR_MODEL].mean().round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════
# 7. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"💾 SAUVEGARDE → data/{VERSION}/")
print("="*70)

keep_cols = [
    'Series_Title', 'Released_Year', 'IMDB_Rating', 'No_of_Votes',
    'Director', 'Genre', 'Genre_Primary', 'Genre_Major', 'Genre_Major_ID',
    'Votes_Log', 'Movie_Age', 'Is_Hidden_Gem',
    *COLS_MM,
]
for star in ['Star1', 'Star2', 'Star3', 'Star4']:
    if star in df_bal.columns:
        keep_cols.append(star)

ml_dataset = df_bal[keep_cols].copy()

ml_dataset.to_csv(os.path.join(DATA_DIR, 'movies_processed_v4.csv'), index=False)
print(f"  ✅ movies_processed_v4.csv")

ml_dataset.to_json(
    os.path.join(DATA_DIR, 'movies_processed_v4.json'),
    orient='records', indent=2
)
print(f"  ✅ movies_processed_v4.json")

genre_mapping = {
    'major_to_id':       MAJOR_GENRE_TO_ID,
    'id_to_major':       {str(k): v for k, v in ID_TO_MAJOR_GENRE.items()},
    'subgenre_to_major': SUBGENRE_TO_MAJOR,
    'genre_groups':      GENRE_GROUPS,
}
with open(os.path.join(DATA_DIR, 'genre_mapping_v4.json'), 'w') as f:
    json.dump(genre_mapping, f, indent=2)
print(f"  ✅ genre_mapping_v4.json")

metadata = {
    'version':             VERSION,
    'total_movies':        len(ml_dataset),
    'features_for_model':  FEATURES_FOR_MODEL,
    'normalized_features': COLS_MM,
    'binary_features':     ['Is_Hidden_Gem'],
    'balancing':           f'undersampling → max {max_allowed} films/classe',
    'votes_median':        float(votes_median),
    'class_distribution':  ml_dataset['Genre_Major'].value_counts().to_dict(),
    'removed_v3_features': [
        'Blockbuster_Score  → corr=1.00 avec No_of_Votes',
        'Rating_x_Votes     → corr=1.00 avec No_of_Votes',
        'Custom_Popularity  → corr=0.91 avec IMDB_Rating',
        'Released_Year_MM   → corr=-1.00 avec Movie_Age',
        'Hidden_Gem_Score   → corr=0.92 avec IMDB_Rating',
        'Recent_Popularity  → redondant',
        'High_Quality       → redondant avec IMDB_Rating',
    ]
}
with open(os.path.join(DATA_DIR, 'metadata_v4.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ metadata_v4.json")

ml_dataset.head(100).to_json(
    os.path.join(DATA_DIR, 'sample_100_movies_v4.json'),
    orient='records', indent=2
)
print(f"  ✅ sample_100_movies_v4.json")

# ══════════════════════════════════════════════════════════════════════════
# 8. RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════
print(f"""
{"="*70}
✅ PREPROCESSING V4 TERMINÉ
{"="*70}

  📊 Films équilibrés   : {len(ml_dataset)}
  🎯 Features modèle    : {FEATURES_FOR_MODEL}
  🔢 Nb features        : {len(FEATURES_FOR_MODEL)}  (au lieu de 11 en V3)

  NORMALISATION :
    ✅ Normalisées   : IMDB_Rating_MM, Votes_Log_MM, Movie_Age_MM
    ✅ Non normalisé : Is_Hidden_Gem (binaire 0/1)
    ❌ V3 normalisait des colonnes déjà construites de colonnes normalisées

  FEATURES SUPPRIMÉES (corrélation > 0.85 en V3) :
    ❌ Blockbuster_Score, Rating_x_Votes → corr=1.00 avec No_of_Votes
    ❌ Custom_Popularity                 → corr=0.91 avec IMDB_Rating
    ❌ Released_Year                     → corr=-1.00 avec Movie_Age
    ❌ Hidden_Gem_Score                  → corr=0.92 avec IMDB_Rating
    ❌ Recent_Popularity, High_Quality   → redondants

⏭️  PROCHAINE ÉTAPE → Naive_Bayes_v4.py :
    X = df[{FEATURES_FOR_MODEL}]
    y = df['Genre_Major_ID']
""")
