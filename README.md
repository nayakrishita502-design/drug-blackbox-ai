# 🧪 AI Drug Black Box Recorder

An AI-powered toxicity prediction system that predicts whether a chemical compound is toxic or non-toxic from its SMILES representation.

## 🚀 Features

- Toxicity prediction using Machine Learning
- Molecular feature extraction using RDKit
- Explainable AI using SHAP
- Streamlit web interface
- Prediction logging system
- Toxicity report generation

## 📂 Project Structure

drug-blackbox-ai/

├── app/
│ ├── app.py
│ └── predict.py
│
├── model/
│ ├── train_model.py
│ └── model.pkl
│
├── utils/
│ ├── features.py
│ └── logger.py
│
├── data/
│ └── tox21.csv
│
├── logs/
│
├── requirements.txt
└── README.md

## 🛠️ Technologies Used

- Python
- Scikit-Learn
- RDKit
- Streamlit
- SHAP
- Pandas

## ▶️ Run Locally

### Clone Repository

```bash
git clone https://github.com/yourusername/drug-blackbox-ai.git
