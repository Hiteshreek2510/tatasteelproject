from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as pipe
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import FileResponse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Feature Engineering Transformer
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Creating new features
        X['failure_frequency'] = X[['TWF', 'HDF', 'PWF', 'OSF']].sum(axis=1)
        X['Wear_per_RPM'] = X['Tool_wear'] / X['Rotational_speed_rpm']
        X['Torque_Tool_Wear_Ratio'] = X['Torque_Nm'] / X['Tool_wear']

        # Drop unwanted columns
        return X


# Define features
cat_features = ['Type']
num_features = ['Air_temperature_K','Process_temperature_K','Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear', 'TWF', 'HDF', 'PWF', 'OSF','RNF']

# Numerical data pipeline
num_pipeline = pipe([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Column Transformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI App
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Input Data Model
class MachineFailureInput(BaseModel):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: int
    Torque_Nm: float
    Tool_wear: int
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int


# Prediction Endpoint
@app.post("/predict")
async def predict_failure(input_data: MachineFailureInput):
    try:
        print("✅ Request received!")  # Check if this prints in PowerShell

        input_dict = input_data.dict()
        print("🔹 Input Data:", input_dict)  # Debugging print

        # Convert dictionary to Pandas DataFrame
        input_df = pd.DataFrame([input_dict],columns=['Type','Air_temperature_K','Process_temperature_K','Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear', 'TWF', 'HDF', 'PWF', 'OSF','RNF'])
        print("📊 DataFrame Created:", input_df)  # Debugging print

        # Transform Data
        pipeline = pipe([('encoder', preprocessor)])
        input_array = pipeline.fit_transform(input_df)
        print("🔢 Transformed Data:", input_array)  # Debugging print

        # Make prediction
        prediction = model.predict(input_array) # Debugging print
        return {"failure_prediction": int(prediction[0])}

    except Exception as e:
        print("❌ Error:", str(e))  # Debugging print
        raise HTTPException(status_code=400, detail=str(e))




# Serve Frontend
@app.get("/")
async def serve_home():
    return FileResponse("frontend.html")


# Run FastAPI
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8080)
