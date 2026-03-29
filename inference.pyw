
import os
import joblib
import numpy as np

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        data = np.array([float(x) for x in request_body.split(",")])
        return data.reshape(1, -1)
    else:
        raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, content_type):
    return str(prediction[0])
