from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
import json

app = FastAPI()

model_infor = {
    "phase-1": {
        "prob-1": {"ckpt_path": "prob-1/prob_1.ckpt"},
        "prob-2": {"ckpt_path": "prob-2/prob_2.ckpt"},
    }
}


def load_model():
    for phase, probs in model_infor.items():
        for prob, prob_info in probs.items():
            model = pickle.load(open(prob_info["ckpt_path"], "rb"))
            model_infor[phase][prob]["model"] = model


load_model()


@app.post("/phase-1/prob-1/predict")
def predict_1_1(data: dict):
    """
    data:
        id: request_id
        columns: 1d-array contains feature names
        rows: 2d-array contains values
    """
    # print(data)
    request_id = data["id"]
    # with open(f"data/prob-1/{request_id}.json", "w") as f:
    #     json.dump(data, f, indent=4)
    model = model_infor["phase-1"]["prob-1"]["model"]
    feature_names = data["columns"]
    feature_values = data["rows"]
    data_df = pd.DataFrame(feature_values, columns=feature_names)
    y_pred = model.predict(data_df)
    output = {}
    output["id"] = request_id
    # output['predictions'] = [0]*len(feature_values)
    output["drift"] = 0
    output["predictions"] = y_pred.tolist()

    return output


@app.post("/phase-1/prob-2/predict")
def predict_1_1(data: dict):
    """
    data:
        id: request_id
        columns: 1d-array contains feature names
        rows: 2d-array contains values
    """
    model = model_infor["phase-1"]["prob-2"]["model"]
    request_id = data["id"]

    # with open(f"data/prob-2/{request_id}.json", "w") as f:
    #     json.dump(data, f, indent=4)
    feature_names = data["columns"]
    feature_values = data["rows"]

    data_df = pd.DataFrame(feature_values, columns=feature_names)
    y_pred = model.predict(data_df)
    output = {}
    output["id"] = request_id
    # output['predictions'] = [0]*len(feature_values)
    output["drift"] = 0
    output["predictions"] = y_pred.tolist()
    return output
