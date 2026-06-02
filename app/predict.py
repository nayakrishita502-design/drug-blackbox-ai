import sys
import os

# Fix path issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from utils.features import smiles_to_features

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Take user input
smiles = input("Enter SMILES: ")

features = smiles_to_features(smiles)

if features is None:
    print("Invalid SMILES ❌")
else:
    prediction = model.predict([features])[0]

    if prediction == 1:
        print("⚠️ Toxic compound")
    else:
        print("✅ Non-toxic compound")