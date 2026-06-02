import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import Counter
import pickle

from utils.features import smiles_to_features

# ================== LOAD DATA ==================
df = pd.read_csv("data/train_dataset_fixed.csv")
print(df.columns)

# ================== TARGET COLUMN ==================
target_column = "SR-p53"

# ================== CLEAN DATA ==================
df = df.dropna(subset=["smiles", target_column])

# ================== CHECK DATA ==================
print("\n--- Target Distribution ---")
print(df[target_column].value_counts())

# ================== PREPARE DATA ==================
X = []
y = []

for _, row in df.iterrows():
    features = smiles_to_features(row['smiles'])

    if features is not None:
        X.append(features)
        y.append(int(row[target_column]))

# ================== CONVERT ==================
X = np.array(X)
y = np.array(y)

print("\nShape:", X.shape)
print("Label distribution:", Counter(y))
print("Unique labels:", set(y))

# ================== OPTIONAL (FASTER TRAINING) ==================
# Reduce size for speed (you can increase later)
X = X[:50000]
y = y[:50000]

# ================== TRAIN TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================== MODEL ==================
model = RandomForestClassifier(
    n_estimators=200,   # reduced for speed
    max_depth=20,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ================== EVALUATION ==================
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================== SAVE MODEL ==================
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model trained and saved!")