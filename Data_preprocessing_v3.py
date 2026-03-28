"""
🎬 DATA PREPROCESSING — V3 : 6226 films  — dataset massif
CSV source : imdb_top_1000_augmente_massif.csv
Sortie     : data/v3/
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import warnings, os
warnings.filterwarnings('ignore')

CSV_FILE  = 'imdb_top_1000_augmente_massif.csv'
VERSION   = 'v3'
LABEL     = '6226 films  — dataset massif'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data', VERSION)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

print("\n" + "🎬"*40)
print(f"  PREPROCESSING {VERSION.upper()} — {LABEL}")
print("🎬"*40 + "\n")

# ── 4 CATÉGORIES MAJEURES ──────────────────────────────────────────────────
GENRE_GROUPS = {
    'Drama':        ['Drama','Biography','Mystery','Film-Noir','Family'],
    'Action':       ['Action','Adventure','Western','Fantasy','Thriller'],
    'Comedy':       ['Comedy','Animation'],
    'Crime_Horror': ['Crime','Horror']
}
SUBGENRE_TO_MAJOR = {sg: m for m, subs in GENRE_GROUPS.items() for sg in subs}
MAJOR_GENRE_TO_ID = {genre: idx for idx, genre in enumerate(GENRE_GROUPS.keys())}
ID_TO_MAJOR_GENRE = {idx: genre for genre, idx in MAJOR_GENRE_TO_ID.items()}

print(f"📌 {len(GENRE_GROUPS)} catégories majeures:")
for major, subgenres in GENRE_GROUPS.items():
    print(f"   [{MAJOR_GENRE_TO_ID[major]}] {major:15} ← {', '.join(subgenres)}")

# ── 1. CHARGEMENT & NETTOYAGE ──────────────────────────────────────────────
print("\n" + "="*80)
print("📥 CHARGEMENT & NETTOYAGE")
print("="*80)

df = pd.read_csv(CSV_FILE)
print(f"✅ Dataset chargé: {df.shape[0]} films")

initial_count = len(df)
df_clean = df.dropna(subset=['Genre']).copy()
df_clean['IMDB_Rating']   = pd.to_numeric(df_clean['IMDB_Rating'],   errors='coerce')
df_clean['Released_Year'] = pd.to_numeric(df_clean['Released_Year'], errors='coerce')
df_clean['No_of_Votes']   = pd.to_numeric(df_clean['No_of_Votes'],   errors='coerce')
df_clean['Gross']         = pd.to_numeric(df_clean['Gross'].replace('', np.nan), errors='coerce')
df_clean = df_clean.dropna(subset=['IMDB_Rating','Released_Year','No_of_Votes'])
df_clean = df_clean.drop_duplicates()
print(f"  ✓ Films nettoyés : {len(df_clean)} / {initial_count}")

# ── 2. GENRES ──────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("🎬 TRAITEMENT DES GENRES")
print("="*80)

df_clean['Genre_Primary']  = df_clean['Genre'].apply(lambda g: 'Unknown' if pd.isna(g) else str(g).split(',')[0].strip())
df_clean['Genres_List']    = df_clean['Genre'].apply(lambda g: [] if pd.isna(g) else [x.strip() for x in str(g).split(',')])
df_clean['Genre_Major']    = df_clean['Genre_Primary'].apply(lambda g: SUBGENRE_TO_MAJOR.get(g,'Drama'))
df_clean['Genre_Major_ID'] = df_clean['Genre_Major'].map(MAJOR_GENRE_TO_ID)
unique_subgenres           = sorted(df_clean['Genre_Primary'].unique())
genre_to_id                = {g: i for i, g in enumerate(unique_subgenres)}
df_clean['Genre_ID']       = df_clean['Genre_Primary'].map(genre_to_id)

print("\n📊 Distribution des 4 catégories:")
for cat, cnt in df_clean['Genre_Major'].value_counts().items():
    pct = cnt / len(df_clean) * 100
    bar = '█' * max(1, cnt // max(1, len(df_clean)//50))
    print(f"  [{MAJOR_GENRE_TO_ID[cat]}] {cat:15} : {cnt:5} films ({pct:.1f}%)  {bar}")

print("\n📊 Sous-genres → catégorie:")
for g, cnt in df_clean['Genre_Primary'].value_counts().items():
    print(f"  {g:15} → {SUBGENRE_TO_MAJOR.get(g,'Drama'):15} : {cnt:5} films")

# ── 3. ENRICHISSEMENT ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("✨ ENRICHISSEMENT DES FEATURES")
print("="*80)

sc = MinMaxScaler()
df_clean['Votes_Score']  = sc.fit_transform(df_clean[['No_of_Votes']])
df_clean['Rating_Score'] = sc.fit_transform(df_clean[['IMDB_Rating']])
df_clean['Gross_Score']  = sc.fit_transform(df_clean[['Gross']].fillna(0))

current_year = 2026
df_clean['Custom_Popularity'] = 0.4*df_clean['Votes_Score'] + 0.4*df_clean['Rating_Score'] + 0.2*df_clean['Gross_Score']
df_clean['Movie_Age']         = current_year - df_clean['Released_Year']
df_clean['Movie_Age_Score']   = sc.fit_transform(df_clean[['Movie_Age']])
df_clean['Blockbuster_Score'] = 0.6*df_clean['Votes_Score'] + 0.4*df_clean['Gross_Score']
df_clean['Hidden_Gem_Score']  = df_clean['Rating_Score'] * (1 - df_clean['Votes_Score'])
df_clean['Votes_Log']         = np.log1p(df_clean['No_of_Votes'])
df_clean['Rating_x_Votes']    = df_clean['IMDB_Rating'] * df_clean['Votes_Score']
df_clean['Recent_Popularity'] = df_clean['Votes_Score'] / (df_clean['Movie_Age'] + 1)
df_clean['High_Quality']      = (df_clean['IMDB_Rating'] > 8).astype(int)
df_clean['Main_Actors']       = df_clean.apply(lambda r: [r['Star1'],r['Star2'],r['Star3'],r['Star4']], axis=1)

for f, d in {'Custom_Popularity':'40% Votes+40% Rating+20% Gross',
              'Blockbuster_Score':'60% Votes+40% Gross',
              'Hidden_Gem_Score':'Rating×(1-Votes)',
              'Votes_Log':'log(1+votes)',
              'Rating_x_Votes':'Rating×Votes',
              'Recent_Popularity':'Votes/(Age+1)',
              'High_Quality':'IMDB>8 (0/1)'}.items():
    print(f"  ✓ {f:22} = {d}")

# ── 4. NORMALISATION MinMax ────────────────────────────────────────────────
print("\n" + "="*80)
print("📐 NORMALISATION MinMax")
print("="*80)

FEATURE_COLS = ['IMDB_Rating','No_of_Votes','Released_Year',
                'Custom_Popularity','Movie_Age','Blockbuster_Score','Hidden_Gem_Score',
                'Votes_Log','Rating_x_Votes','Recent_Popularity','High_Quality']

scaler_mm = MinMaxScaler()
X_mm      = scaler_mm.fit_transform(df_clean[FEATURE_COLS])
mm_cols   = [f + '_MM' for f in FEATURE_COLS]
df_clean[mm_cols] = X_mm
df_clean['IMDB_Rating_Normalized'] = df_clean['IMDB_Rating_MM']
df_clean['Votes_Normalized']       = df_clean['No_of_Votes_MM']
df_clean['Year_Normalized']        = df_clean['Released_Year_MM']
df_clean['Popularity_Normalized']  = df_clean['Custom_Popularity_MM']

for col, orig in zip(mm_cols, FEATURE_COLS):
    print(f"  ✓ {orig:25} → {col}  [{df_clean[col].min():.3f}, {df_clean[col].max():.3f}]")

# ── 5. DATASET FINAL & SAUVEGARDE ─────────────────────────────────────────
print("\n" + "="*80)
print(f"💾 SAUVEGARDE → data/{VERSION}/")
print("="*80)

keep_cols = [
    'Series_Title','Released_Year','IMDB_Rating','No_of_Votes',
    'Director','Star1','Star2','Star3','Star4',
    'Genre','Genre_Primary','Genres_List','Genre_ID',
    'Genre_Major','Genre_Major_ID',
    'Gross','Custom_Popularity','Movie_Age',
    'Blockbuster_Score','Hidden_Gem_Score',
    'Votes_Log','Rating_x_Votes','Recent_Popularity','High_Quality',
    *mm_cols,
    'IMDB_Rating_Normalized','Votes_Normalized',
    'Year_Normalized','Popularity_Normalized','Movie_Age_Score'
]
ml_dataset      = df_clean[keep_cols].copy()
ml_dataset_save = ml_dataset.copy()
ml_dataset_save['Genres_List'] = ml_dataset_save['Genres_List'].apply(list)

ml_dataset_save.to_csv(os.path.join(DATA_DIR,'movies_processed_enriched.csv'), index=False)
ml_dataset_save.to_json(os.path.join(DATA_DIR,'movies_processed_enriched.json'), orient='records', indent=2)
print(f"  ✅ data/{VERSION}/movies_processed_enriched.csv")
print(f"  ✅ data/{VERSION}/movies_processed_enriched.json")

genre_mapping = {
    'genre_to_id': genre_to_id,
    'id_to_genre': {str(k): v for k,v in enumerate(unique_subgenres)},
    'major_to_id': MAJOR_GENRE_TO_ID,
    'id_to_major': {str(k): v for k,v in ID_TO_MAJOR_GENRE.items()},
    'subgenre_to_major': SUBGENRE_TO_MAJOR,
    'genre_groups': GENRE_GROUPS,
    'total_subgenres': len(genre_to_id),
    'total_major': len(MAJOR_GENRE_TO_ID)
}
with open(os.path.join(DATA_DIR,'genre_mapping.json'),'w') as f:
    json.dump(genre_mapping, f, indent=2)
print(f"  ✅ data/{VERSION}/genre_mapping.json")

metadata = {
    'version': VERSION, 'label': LABEL, 'csv_source': CSV_FILE,
    'total_movies': len(ml_dataset), 'features_ml': mm_cols,
    'class_distribution': ml_dataset['Genre_Major'].value_counts().to_dict()
}
with open(os.path.join(DATA_DIR,'metadata_enriched.json'),'w') as f:
    json.dump(metadata, f, indent=2)
ml_dataset_save.head(100).to_json(os.path.join(DATA_DIR,'sample_100_movies_enriched.json'), orient='records', indent=2)
print(f"  ✅ data/{VERSION}/metadata_enriched.json")
print(f"  ✅ data/{VERSION}/sample_100_movies_enriched.json")

print(f"""
{"="*80}
✅ PREPROCESSING {VERSION.upper()} TERMINÉ — {LABEL}
{"="*80}
  Films     : {len(ml_dataset)}
  Features  : {len(FEATURE_COLS)} colonnes MinMax
  Dossier   : data/{VERSION}/
""")