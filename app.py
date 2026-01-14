# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo guardado
model = joblib.load("california_housing_model.pkl")

# Inicializar FastAPI
app = FastAPI(title="Predicción de Precios de Viviendas en California")


# Definir el esquema de entrada
class HouseData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.post("/predict")
def predict_price(data: HouseData):
    # Convertir a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Predecir
    prediction = model.predict(input_data)[0]

    return {"predicted_median_house_value": round(prediction, 2)}