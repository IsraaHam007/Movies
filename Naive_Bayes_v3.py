"""
🤖 NAIVE BAYES — V3 : 6226 films  — dataset massif
Prérequis : data_preprocessing_v3.py doit avoir été lancé avant.
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

VERSION    = 'v3'
LABEL      = '6226 films  — dataset massif'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data',   VERSION)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', VERSION)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

print("\n" + "🤖"*40)
print(f"  NAIVE BAYES — {VERSION.upper()} : {LABEL}")
print("🤖"*40 + "\n")

# ── CHARGEMENT ─────────────────────────────────────────────────────────────
csv_file = os.path.join(DATA_DIR, 'movies_processed_enriched.csv')
if not os.path.exists(csv_file):
    print(f"❌ ERREUR: Lancez d\'abord data_preprocessing_{VERSION}.py")
    exit(1)

df = pd.read_csv(csv_file)
print(f"✅ {len(df)} films chargés")

with open(os.path.join(DATA_DIR,'genre_mapping.json')) as f:
    genre_mapping = json.load(f)

CLASS_NAMES = list(genre_mapping['major_to_id'].keys())
MAJOR_TO_ID = genre_mapping['major_to_id']
ID_TO_MAJOR = {int(k): v for k,v in genre_mapping['id_to_major'].items()}

print(f"\n📌 Classes : {CLASS_NAMES}")
print("\n📊 Distribution:")
for cat, cnt in df['Genre_Major'].value_counts().items():
    pct = cnt/len(df)*100
    print(f"   {cat:15} : {cnt:5} films ({pct:.1f}%)")

# ── FEATURES ───────────────────────────────────────────────────────────────
TARGET = 'Genre_Major_ID'
y = df[TARGET].values

FEATURE_COLS = [
    'IMDB_Rating_MM','No_of_Votes_MM','Released_Year_MM',
    'Custom_Popularity_MM','Blockbuster_Score_MM','Hidden_Gem_Score_MM',
    'Votes_Log_MM','Rating_x_Votes_MM','Recent_Popularity_MM','High_Quality_MM'
]
print(f"\n📐 {len(FEATURE_COLS)} features MinMax utilisées")

# ── CROSS-VALIDATION ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("📐 CROSS-VALIDATION (GaussianNB + MinMax)")
print("="*80)

X = df[FEATURE_COLS].values
cv5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores_5  = cross_val_score(GaussianNB(), X, y, cv=cv5,  scoring='accuracy')
scores_10 = cross_val_score(GaussianNB(), X, y, cv=cv10, scoring='accuracy')

print(f"\n  5-fold  → Accuracy: {scores_5.mean():.4f}  (±{scores_5.std():.4f})")
print(f"            Scores  : {[round(s,4) for s in scores_5]}")
print(f"\n  10-fold → Accuracy: {scores_10.mean():.4f}  (±{scores_10.std():.4f})")
print(f"            Scores  : {[round(s,4) for s in scores_10]}")

cv_results = {
    'cv5_mean': float(scores_5.mean()),  'cv5_std': float(scores_5.std()),
    'cv5_scores': [round(float(s),4) for s in scores_5],
    'cv10_mean': float(scores_10.mean()), 'cv10_std': float(scores_10.std()),
    'cv10_scores': [round(float(s),4) for s in scores_10]
}

# ── ENTRAÎNEMENT FINAL ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("🎓 ENTRAÎNEMENT FINAL")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  ✓ Train : {len(X_train)} | Test : {len(X_test)}")

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── MÉTRIQUES GLOBALES ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("📈 MÉTRIQUES GLOBALES")
print("="*80)

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test,    y_pred, average='weighted', zero_division=0)
f1  = f1_score(y_test,        y_pred, average='weighted', zero_division=0)

print(f"\n  ✓ Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  ✓ Precision: {pre:.4f}")
print(f"  ✓ Recall   : {rec:.4f}")
print(f"  ✓ F1-Score : {f1:.4f}")

# ── MÉTRIQUES PAR CATÉGORIE ────────────────────────────────────────────────
print("\n" + "="*80)
print("📊 MÉTRIQUES PAR CATÉGORIE")
print("="*80)

report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

print(f"  {'Catégorie':<18} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8} {'Support':>9}")
print(f"  {'-'*68}")

per_class_metrics = {}
for cat in CLASS_NAMES:
    if cat in report_dict:
        r = report_dict[cat]
        cat_id    = MAJOR_TO_ID[cat]
        mask_true = (y_test == cat_id)
        mask_pred = (y_pred == cat_id)
        tp = (mask_true  & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true  & ~mask_pred).sum()
        tn = (~mask_true & ~mask_pred).sum()
        cat_acc = (tp + tn) / len(y_test)
        per_class_metrics[cat] = {
            'accuracy': float(cat_acc), 'precision': float(r['precision']),
            'recall': float(r['recall']), 'f1_score': float(r['f1-score']),
            'support': int(r['support']),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        }
        print(f"  {cat:<18} {cat_acc:>10.4f} {r['precision']:>11.4f} {r['recall']:>8.4f} {r['f1-score']:>8.4f} {int(r['support']):>9}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Matrice de confusion:")
print(f"  {'':18}" + "".join(f"{n:>14}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:<18}" + "".join(f"{v:>14}" for v in row))

best_cat  = max(per_class_metrics, key=lambda k: per_class_metrics[k]['f1_score'])
worst_cat = min(per_class_metrics, key=lambda k: per_class_metrics[k]['f1_score'])
print(f"\n  ✅ Meilleure : {best_cat}  (F1={per_class_metrics[best_cat]['f1_score']:.4f})")
print(f"  ⚠️  Pire     : {worst_cat} (F1={per_class_metrics[worst_cat]['f1_score']:.4f})")

# ── SAUVEGARDE ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"💾 SAUVEGARDE → models/{VERSION}/")
print("="*80)

model_path = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

model_info = {
    'model_type': 'GaussianNB', 'version': VERSION, 'label': LABEL,
    'normalization': 'MinMax', 'features': FEATURE_COLS,
    'num_classes': len(CLASS_NAMES), 'class_names': CLASS_NAMES,
    'genre_mapping': genre_mapping, 'cv_results': cv_results,
    'global_metrics': {'accuracy': float(acc), 'precision': float(pre),
                        'recall': float(rec), 'f1_score': float(f1)},
    'per_class_metrics': per_class_metrics
}
info_path = os.path.join(MODELS_DIR, 'model_info.json')
with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"  ✅ models/{VERSION}/naive_bayes_model.pkl")
print(f"  ✅ models/{VERSION}/model_info.json")

print(f"""
{"="*80}
✅ NAIVE BAYES {VERSION.upper()} — {LABEL}
  Accuracy : {acc:.4f}  |  F1 : {f1:.4f}
  CV5      : {scores_5.mean():.4f} ±{scores_5.std():.4f}
{"="*80}
""")

# ── API FLASK ──────────────────────────────────────────────────────────────
print("\n" + "🚀"*40)
print(f"  API FLASK — {VERSION.upper()}")
print("🚀"*40 + "\n")

app = Flask(__name__)
CORS(app)

with open(model_path,'rb') as f: TRAINED_MODEL = pickle.load(f)
with open(info_path) as f:       MODEL_INFO    = json.load(f)
df_api          = pd.read_csv(csv_file)
GENRE_MAPPING   = MODEL_INFO['genre_mapping']
CLASS_NAMES_API = MODEL_INFO['class_names']
print(f"✅ {len(df_api)} films | Classes : {CLASS_NAMES_API}\n")

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','version':VERSION,'label':LABEL,
                    'model':'GaussianNB','movies_in_db':len(df_api),
                    'accuracy':MODEL_INFO['global_metrics']['accuracy']})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({'cv_results':MODEL_INFO['cv_results'],
                    'global_metrics':MODEL_INFO['global_metrics'],
                    'per_class_metrics':MODEL_INFO['per_class_metrics']})

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data       = request.json
        min_rating = float(data.get('min_rating', 7.0))
        era        = data.get('era','mixed')
        niche      = data.get('niche', False)
        director   = data.get('favorite_director','').lower()
        genre_pref = data.get('genre_preference','').lower()

        filtered = df_api[df_api['IMDB_Rating'] >= min_rating].copy()
        if era == 'modern':  filtered = filtered[filtered['Released_Year'] >= 2000]
        elif era == 'vintage': filtered = filtered[filtered['Released_Year'] < 2000]
        med = df_api['No_of_Votes'].median()
        filtered = filtered[filtered['No_of_Votes'] < med] if niche else filtered[filtered['No_of_Votes'] >= med]
        if genre_pref:
            filtered = filtered[
                filtered['Genre_Major'].str.lower().str.contains(genre_pref, na=False) |
                filtered['Genre_Primary'].str.lower().str.contains(genre_pref, na=False)]
        if director:
            filtered = filtered[filtered['Director'].str.lower().str.contains(director, na=False)]

        top = filtered.nlargest(10,'IMDB_Rating')
        recs = [{'title':r['Series_Title'],'year':int(r['Released_Year']),
                  'rating':float(r['IMDB_Rating']),'votes':int(r['No_of_Votes']),
                  'genre':r['Genre_Primary'],'major_genre':r['Genre_Major'],
                  'director':r['Director']} for _,r in top.iterrows()]
        return jsonify({'version':VERSION,'total_results':len(recs),'recommendations':recs})
    except Exception as e:
        return jsonify({'error':str(e)}), 400

@app.route('/api/genres',  methods=['GET'])
def get_genres():  return jsonify({'major_genres':CLASS_NAMES_API,'sub_genres':sorted(GENRE_MAPPING['genre_to_id'].keys()),'genre_groups':GENRE_MAPPING['genre_groups']})

@app.route('/api/stats',   methods=['GET'])
def get_stats():   return jsonify({'version':VERSION,'total_movies':len(df_api),'model_metrics':MODEL_INFO['global_metrics'],'per_class':MODEL_INFO['per_class_metrics']})

if __name__ == '__main__':
    print(f"\n🚀 API {VERSION.upper()} → http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)