
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cargar datos (siempre desde la URL)
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Preprocesamiento
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Entrenar modelo
print("✅ Entrenando modelo... (esto toma unos segundos)")
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model_pipeline.fit(X, y)
print("✅ Modelo entrenado!")

# Inicializar FastAPI
app = FastAPI(title="Predicción de Precios de Viviendas en California")

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
    input_data = pd.DataFrame([data.dict()])
    prediction = model_pipeline.predict(input_data)[0]
    return {"predicted_median_house_value": round(prediction, 2)}
