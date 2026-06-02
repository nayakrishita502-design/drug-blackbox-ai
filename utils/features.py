# from rdkit import Chem
# from rdkit.Chem import Descriptors

# def smiles_to_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)

#     if mol is None:
#         return None

#     features = {
#         "MolWt": Descriptors.MolWt(mol),
#         "LogP": Descriptors.MolLogP(mol),
#         "NumHDonors": Descriptors.NumHDonors(mol),
#         "NumHAcceptors": Descriptors.NumHAcceptors(mol)
#     }

#     return list(features.values())

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_features(smiles):
    # Convert SMILES → molecule
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None

    # Generate Morgan Fingerprint (2048 features)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    # Convert to numpy array
    arr = np.zeros((2048,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)


    return arr