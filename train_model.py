"""
AI-Based Flood Prediction and Detection System
Model Training Script
Trains multiple ML models and saves the best one with accuracy metrics.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Data ──────────────────────────────────────────────
DATA_FILE = "Flood prediction System.csv"
data = pd.read_csv(DATA_FILE)

print("=== Dataset Info ===")
print(f"Shape: {data.shape}")
print(data.head())
print(f"\nMissing values:\n{data.isnull().sum()}")
print(f"\nClass distribution:\n{data['flood'].value_counts()}")

# ── 2. Feature / Target Split ─────────────────────────────────
FEATURES = ["rainfall", "water_level", "humidity", "temperature"]
TARGET = "flood"

X = data[FEATURES]
y = data[TARGET]

# ── 3. Train / Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scaler ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── 5. Train Models ───────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
best_model = None
best_name = ""
best_acc = 0

print("\n=== Model Training Results ===")
for name, model in models.items():
    # Train (use scaled data only for Logistic Regression)
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "cv_mean": round(cv_scores.mean() * 100, 2),
        "cv_std": round(cv_scores.std() * 100, 2),
    }

    print(f"\n{name}:")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(f"  CV Score : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\n>>> Best Model: {best_name} ({best_acc*100:.2f}% accuracy)")

# ── 6. Feature Importances ────────────────────────────────────
feature_importances = {}

# Use Random Forest for feature importance (most reliable)
rf = models["Random Forest"]
importances = rf.feature_importances_
for feat, imp in zip(FEATURES, importances):
    feature_importances[feat] = round(float(imp) * 100, 2)

print(f"\n=== Feature Importances (Random Forest) ===")
for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.2f}%")

# ── 7. Confusion Matrix Plot ──────────────────────────────────
os.makedirs("static/img", exist_ok=True)

if best_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_sc)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('#0a0e1a')
ax.set_facecolor('#111827')
sns.heatmap(
    cm,
    annot=True, fmt='d',
    cmap='Blues',
    xticklabels=['No Flood', 'Flood'],
    yticklabels=['No Flood', 'Flood'],
    ax=ax,
    linewidths=0.5,
    linecolor='#1e3a5f'
)
ax.set_title(f'Confusion Matrix - {best_name}', color='#e2e8f0', pad=15, fontsize=13)
ax.set_xlabel('Predicted', color='#94a3b8', labelpad=10)
ax.set_ylabel('Actual', color='#94a3b8', labelpad=10)
ax.tick_params(colors='#94a3b8')
plt.tight_layout()
plt.savefig('static/img/confusion_matrix.png', dpi=100, bbox_inches='tight',
            facecolor='#0a0e1a', edgecolor='none')
plt.close()

# ── 8. Feature Importance Plot ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor('#0a0e1a')
ax.set_facecolor('#111827')

sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
feat_names = [f[0].replace('_', '\n') for f in sorted_feats]
feat_vals = [f[1] for f in sorted_feats]
colors = ['#00d4ff', '#0099bb', '#006688', '#004455']

bars = ax.barh(feat_names, feat_vals, color=colors[:len(feat_names)], height=0.5)
ax.set_xlabel('Importance (%)', color='#94a3b8')
ax.set_title('Feature Importances', color='#e2e8f0', pad=15, fontsize=13)
ax.tick_params(colors='#94a3b8')
ax.spines['bottom'].set_color('#1e3a5f')
ax.spines['left'].set_color('#1e3a5f')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, val in zip(bars, feat_vals):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', color='#94a3b8', fontsize=10)

plt.tight_layout()
plt.savefig('static/img/feature_importance.png', dpi=100, bbox_inches='tight',
            facecolor='#0a0e1a', edgecolor='none')
plt.close()

# ── 9. Correlation Heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#0a0e1a')
ax.set_facecolor('#111827')

corr = data.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = False

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    corr, annot=True, fmt='.2f', cmap=cmap,
    ax=ax, linewidths=0.5, linecolor='#1e3a5f',
    vmin=-1, vmax=1,
    annot_kws={'size': 9, 'color': '#e2e8f0'}
)
ax.set_title('Feature Correlation Heatmap', color='#e2e8f0', pad=15, fontsize=13)
ax.tick_params(colors='#94a3b8')
plt.tight_layout()
plt.savefig('static/img/correlation_heatmap.png', dpi=100, bbox_inches='tight',
            facecolor='#0a0e1a', edgecolor='none')
plt.close()

# ── 10. Save Model & Metadata ─────────────────────────────────
# For consistency save both the best model and scaler separately
model_data = {
    "model": best_model,
    "scaler": scaler,
    "model_name": best_name,
    "features": FEATURES,
    "needs_scaling": best_name == "Logistic Regression"
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Save metadata as JSON for API
metadata = {
    "best_model": best_name,
    "accuracy": results[best_name]["accuracy"],
    "features": FEATURES,
    "all_models": results,
    "feature_importances": feature_importances,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "total_samples": len(data),
    "flood_rate": round(data['flood'].mean() * 100, 1)
}

os.makedirs("static", exist_ok=True)
with open("static/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n[OK] Model saved to model.pkl")
print("[OK] Metadata saved to static/model_metadata.json")
print("[OK] Plots saved to static/img/")
print("\n=== All Models Summary ===")
for model_name, metrics in results.items():
    print(f"  {model_name}: {metrics['accuracy']}% accuracy")
