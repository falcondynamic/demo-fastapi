from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

rf_model=joblib.load('pipeline_model_rf')


@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/prediction/{param1}/{param2}/{param3}/{param4}/{param5}/{param6}")
def predict_price(param1: int, param2: int, param3: str, param4: int, param5: str, param6: str):
    my_dict = {
        "gabel_federweg_mm": param1,
        "akkukapazität_wh": param2,
        "rahmenmaterial": param3,
        'gänge': param4,
        "kategorie": param5,
        "hersteller": param6    
    }

    df = pd.DataFrame.from_dict([my_dict])

    prediction = rf_model.predict(df)

    return {"prediction": int(prediction[0]), "unit":"€", "message": "Prediction successful!"}
