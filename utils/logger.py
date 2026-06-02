import json
from datetime import datetime

def log_prediction(smiles, result):
    log = {
        "smiles": smiles,
        "result": result,
        "time": str(datetime.now())
    }

    try:
        with open("logs/logs.json", "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(log)

    with open("logs/logs.json", "w") as f:
        json.dump(data, f, indent=4)