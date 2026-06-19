import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "model_stress_terbaik.pkl"
)

model = joblib.load(MODEL_PATH)

def predict_stress(data):
    df = pd.DataFrame([data])

    pred = model.predict(df)[0]

    return "tinggi" if pred == 1 else "rendah"


def get_focus(attention):
    if attention >= 60:
        return "tinggi"
    else:
        return "rendah"