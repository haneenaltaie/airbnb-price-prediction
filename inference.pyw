import os
import joblib
import numpy as np
import json

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        values = [float(x) for x in request_body.strip().split(",")]
        return np.array(values).reshape(1, -1)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Model was trained on log price, convert back to dollars
    log_pred = model.predict(input_data)
    return np.expm1(log_pred)

def output_fn(prediction, accept):
    return str(float(prediction[0]))
