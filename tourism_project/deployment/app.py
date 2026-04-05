
import os
import uvicorn
import joblib
import pandas as pd
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from multiprocessing import Process
from huggingface_hub import hf_hub_download

# --- 1. GLOBAL CONFIGURATION ---
REPO_ID = "GauthamJ007/VisitWithUs-Wellness-Tourism-Predictor"
FILENAME = "tourism_pipeline.joblib"

# --- 2. BACKEND: FastAPI Setup ---
app = FastAPI(title="Visit with Us - Backend API")

class CustomerData(BaseModel):
    Age: float
    TypeofContact: str
    CityTier: int
    DurationOfPitch: float
    Occupation: str
    Gender: str
    NumberOfPersonVisiting: int
    NumberOfFollowups: float
    ProductPitched: str
    PreferredPropertyStar: float
    MaritalStatus: str
    NumberOfTrips: float
    Passport: int
    PitchSatisfactionScore: int
    OwnCar: int
    NumberOfChildrenVisiting: float
    Designation: str
    MonthlyIncome: float

@st.cache_resource
def load_model_from_hf():
    """Downloads and caches the model artifact from Hugging Face Hub."""
    token = os.getenv("HF_TOKEN")
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            token=token
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"🚨 Model Load Error: {e}")
        st.info("Ensure HF_TOKEN is in 'Secrets' and REPO_ID/FILENAME are correct.")
        return None

@app.post("/predict_api")
def predict_api(data: CustomerData):
    model = load_model_from_hf()
    if model is None:
        return {"error": "Model not loaded"}

    input_df = pd.DataFrame([data.dict()])
    # --- MLOPS FIX: Satisfy Pipeline metadata requirement ---
    input_df['__index_level_0__'] = 0
    
    # --- FIX: Extract scalar values from NumPy arrays ---
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- 3. FRONTEND: Streamlit UI ---
def run_ui():
    st.set_page_config(page_title="Visit with Us | Wellness Predictor", layout="wide")

    st.title("🏝️ Wellness Package Predictor")
    st.info("Lead Scoring System for the 'Visit with Us' Sales Team")

    model = load_model_from_hf()
    if model is None:
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Customer Profile")
            age = st.number_input("Age", 18, 80, 35)
            income = st.number_input("Monthly Income", 1000, 100000, 25000)
            passport = st.radio("Has Passport?",[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            occ = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

        with col2:
            st.subheader("📞 Interaction Data")
            pitch_dur = st.slider("Pitch Duration (Mins)", 5, 120, 15)
            followups = st.slider("Number of Follow-ups", 1, 6, 3)
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            stars = st.selectbox("Preferred Hotel Stars", [1,2,3,4,5])
            own_car = st.radio("Owns Car?",[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        submit = st.form_submit_button("Predict Lead Potential")

    if submit:
        # Build raw dictionary exactly matching training features
        input_dict = {
            "Age": age,
            "TypeofContact": "Self Enquiry",
            "CityTier": 1,
            "DurationOfPitch": pitch_dur,
            "Occupation": occ,
            "Gender": gender,
            "NumberOfPersonVisiting": 2,
            "NumberOfFollowups": followups,
            "ProductPitched": "Basic",
            "PreferredPropertyStar": float(stars),
            "MaritalStatus": marital,
            "NumberOfTrips": 3.0,
            "Passport": passport,
            "PitchSatisfactionScore": 3,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": 1.0,
            "Designation": designation,
            "MonthlyIncome": float(income)
        }

        input_df = pd.DataFrame([input_dict])
        
        # --- MLOPS FIX: Add missing index column to match training schema ---
        input_df['__index_level_0__'] = 0

        try:
            prob = model.predict_proba(input_df)[0][1]
            st.divider()

            if prob >= 0.45:
                msg = f"### 🔥 High Potential Lead to buy Wellness Package!\n**Purchase Probability:** {prob:.2%}"
                st.success(msg)
                st.balloons()
            else:
                msg = f"### 🧊 Low Potential Lead may not buy Wellness Package.\n**Purchase Probability:** {prob:.2%}"
                st.warning(msg)
        except Exception as e:
            st.error(f"Inference Error: {e}")

# --- 4. MLOPS EXECUTION FLOW ---
if __name__ == "__main__":
    # Start API in background
    api_process = Process(target=run_api, daemon=True)
    api_process.start()
    
    # Run Streamlit
    run_ui()
