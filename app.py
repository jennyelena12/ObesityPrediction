# Elena Nathanielle Budiman Angkawi - 2702213764

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# judul
st.set_page_config(
    page_title="Obesity Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# konek dengan API endpoint kita 
API_URL = "http://localhost:8000"

# CSS style untuk FE
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(data):
    """Make prediction using the API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def main():
    # Header
    st.markdown('<div class="main-header">🏥 Obesity Prediction System</div>', unsafe_allow_html=True)
    
    # cek apakah sudah terkonek
    if not check_api_health():
        st.error("❌ API is not running. Please start the FastAPI backend server.")
        st.info("Run the following command in your terminal: `uvicorn main:app --reload`")
        return
    
    st.success("✅ API is running successfully!")
    
    # dapatkan model yang sudah kita save
    model_info = get_model_info()
    if not model_info:
        st.error("Could not retrieve model information from API")
        return
    
    st.sidebar.header("📊 Input Parameters")
    # input user
    with st.sidebar:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", model_info["gender_options"])
        age = st.slider("Age", min_value=10, max_value=100, value=25)
        height = st.slider("Height (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
        weight = st.slider("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        
        st.subheader("Family & Lifestyle")
        family_history = st.selectbox("Family History with Overweight", model_info["yes_no_options"])
        favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", model_info["yes_no_options"])
        smoke = st.selectbox("Smoke", model_info["yes_no_options"])
        scc = st.selectbox("Calories consumption monitoring (SCC)", model_info["yes_no_options"])
        
        st.subheader("Eating Habits")
        fcvc = st.slider("Frequency of consumption of vegetables (FCVC)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        ncp = st.slider("Number of main meals (NCP)", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
        caec = st.selectbox("Consumption of food between meals (CAEC)", model_info["caec_options"])
        ch2o = st.slider("Consumption of water daily (CH2O)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        calc = st.selectbox("Consumption of alcohol (CALC)", model_info["calc_options"])
        
        st.subheader("Physical Activity")
        faf = st.slider("Physical activity frequency (FAF)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        tue = st.slider("Time using technology devices (TUE)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        mtrans = st.selectbox("Transportation used (MTRANS)", model_info["transport_options"])
        
        predict_button = st.button("🔮 Make Prediction", use_container_width=True)
    

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("📈 Prediction Results")
        
        if predict_button:
            # masukkan data dari user
            data = {
                "Gender": gender,
                "Age": age,
                "Height": height,
                "Weight": weight,
                "family_history_with_overweight": family_history,
                "FAVC": favc,
                "FCVC": fcvc,
                "NCP": ncp,
                "CAEC": caec,
                "SMOKE": smoke,
                "CH2O": ch2o,
                "SCC": scc,
                "FAF": faf,
                "TUE": tue,
                "CALC": calc,
                "MTRANS": mtrans
            }
            
            # buat prediksi
            with st.spinner("Making prediction..."):
                result = make_prediction(data)
            if result:
                prediction = result["prediction"]
                confidence = result["confidence"]
                bmi = result["bmi"]
                bmi_category = result["bmi_category"]
                
                # estetika per kategori
                if prediction in ["Normal_Weight", "Insufficient_Weight"]:
                    box_class = "success-box"
                    status_icon = "✅"
                elif prediction in ["Overweight_Level_I", "Overweight_Level_II"]:
                    box_class = "warning-box"
                    status_icon = "⚠️"
                else:
                    box_class = "danger-box"
                    status_icon = "🚨"
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>{status_icon} Prediction: {prediction.replace('_', ' ')}</h2>
                    <h3>Confidence: {confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # cek user input (utk debug)
                st.subheader("Input Summary")
                input_df = pd.DataFrame([data]).T
                input_df.columns = ["Value"]
                st.dataframe(input_df, use_container_width=True)
                
        else:
            st.info("👈 Please fill in the parameters in the sidebar and click 'Make Prediction'")
    
    with col2:
        st.header("ℹ️ Information")
        

        # target variabel yg di predict
        st.subheader("Prediction Classes")
        prediction_classes = [
            "Insufficient Weight",
            "Normal Weight", 
            "Overweight Level I",
            "Overweight Level II",
            "Obesity Type I",
            "Obesity Type II", 
            "Obesity Type III"
        ]
        
        for i, class_name in enumerate(prediction_classes):
            st.text(f"{i}: {class_name}")
        
        # tambahan tips
        st.subheader("💡 Health Tips")
        st.markdown("""
        - Maintain a balanced diet
        - Exercise regularly
        - Stay hydrated
        - Monitor your weight
        - Avoid excessive alcohol
        - Get adequate sleep
        """)

if __name__ == "__main__":
    main()