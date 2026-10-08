# Elena Nathanielle Budiman Angkawi - 2702213764

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# inisialisasi fastAPI
app = FastAPI(title="Obesity Prediction API", version="1.0.0")

# tambah CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
try:
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: xgb_model.pkl not found. Please ensure the model file is in the same directory.")
    model = None

# sesuaikan mappings sesuai encode kita
gender_mapping = {'Male': 0, 'Female': 1}
yes_no_mapping = {'yes': 1, 'no': 0}
caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
mtrans_mapping = {
    'Public_Transportation': 0,
    'Automobile': 1,
    'Walking': 2,
    'Motorbike': 3,
    'Bike': 4
}
nobeyesdad_mapping = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}

reverse_nobeyesdad_mapping = {v: k for k, v in nobeyesdad_mapping.items()}

# define variabel2 nya
class PredictionRequest(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

# define response model
class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: int
    confidence: float
    bmi: float
    bmi_category: str

def calculate_bmi_category(bmi: float) -> str:
    """Calculate BMI category based on BMI value"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

@app.get("/")
async def root():
    return {"message": "Obesity Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_obesity(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # ubah input sesuai format yg diinginkan model
        data = {
            'Gender': gender_mapping.get(request.Gender),
            'Age': request.Age,
            'Height': request.Height,
            'Weight': request.Weight,
            'family_history_with_overweight': yes_no_mapping.get(request.family_history_with_overweight),
            'FAVC': yes_no_mapping.get(request.FAVC),
            'FCVC': request.FCVC,
            'NCP': request.NCP,
            'CAEC': caec_mapping.get(request.CAEC),
            'SMOKE': yes_no_mapping.get(request.SMOKE),
            'CH2O': request.CH2O,
            'SCC': yes_no_mapping.get(request.SCC),
            'FAF': request.FAF,
            'TUE': request.TUE,
            'CALC': calc_mapping.get(request.CALC),
            'MTRANS': mtrans_mapping.get(request.MTRANS)
        }
        
        # debug jika ada invalidation
        if None in data.values():
            raise HTTPException(status_code=400, detail="Invalid input values")
        
        df = pd.DataFrame([list(data.values())], columns=list(data.keys()))
        
        # buat prediksi
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        # Calculate BMI
        bmi = request.Weight / (request.Height ** 2)
        bmi_category = calculate_bmi_category(bmi)
        
        # confidence score
        confidence = float(prediction_proba[prediction])
        
        # prediction label
        prediction_label = reverse_nobeyesdad_mapping.get(prediction, "Unknown")
        
        return PredictionResponse(
            prediction=prediction_label,
            prediction_code=int(prediction),
            confidence=confidence,
            bmi=round(bmi, 2),
            bmi_category=bmi_category
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the model and expected inputs"""
    return {
        "model_type": "XGBoost",
        "features": [
            "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
            "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
        ],
        "gender_options": list(gender_mapping.keys()),
        "yes_no_options": list(yes_no_mapping.keys()),
        "caec_options": list(caec_mapping.keys()),
        "calc_options": list(calc_mapping.keys()),
        "transport_options": list(mtrans_mapping.keys()),
        "prediction_classes": list(nobeyesdad_mapping.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)