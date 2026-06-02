from utils.features import smiles_to_features

smiles = "CCO"
features = smiles_to_features(smiles)

print("Features:", features)