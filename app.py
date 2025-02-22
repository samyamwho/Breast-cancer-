import streamlit as st
import joblib
import numpy as np
import os

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load a model safely using joblib
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {model_path}. {str(e)}")
        return None

# Define model filenames (these models must be retrained on the 10 individual features)
model_files = {
    "XGBoost": "XGBoost_10.pkl",
    "Linear Regression": "LinearRegression_10.pkl",
    "Elastic Net": "ElasticNet_10.pkl",
    "MLP Regression":"MLPReggressir_10.pkl"
    # Add additional models as available
}

# Load models dynamically
models = {}
for model_name, file_name in model_files.items():
    model_path = os.path.join(current_dir, file_name)
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is not None:
            models[model_name] = model
    else:
        st.warning(f"⚠️ Model '{model_name}' not found! Skipping...")

st.title("🔬 Breast Cancer Survival Prediction")
st.markdown("### **Select a model and enter your details to predict your survival rate**")

if len(models) == 0:
    st.error("No models available. Please retrain and save models with the individual-level features.")
    st.stop()

selected_model_name = st.selectbox("🛠 Choose a Prediction Model:", list(models.keys()))
selected_model = models[selected_model_name]

if not hasattr(selected_model, "predict"):
    st.error("❌ Error: The selected model is not valid. Please choose another model.")
    st.stop()

st.write("### **📊 Enter Your Personal & Health Details**")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input(
        "💰 Annual Income (USD)",
        min_value=0.0,
        max_value=500000.0,
        step=1000.0,
        help="Your total annual income in US dollars."
    )
    annual_healthcare = st.number_input(
        "🏥 Annual Healthcare Spending (USD)",
        min_value=0.0,
        max_value=50000.0,
        step=100.0,
        help="How much you spend on healthcare per year."
    )
    screening_years = st.number_input(
        "📅 Years in Screening Programs",
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        help="Number of years you've participated in regular screening."
    )
    access_rating = st.slider(
        "🩺 Access to Care Rating",
        0,
        100,
        50,
        help="On a scale from 0 (poor) to 100 (excellent), rate your access to healthcare."
    )
    weight = st.number_input(
        "⚖️ Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        step=0.5,
        help="Your weight in kilograms."
    )

with col2:
    height = st.number_input(
        "📏 Height (cm)",
        min_value=100.0,
        max_value=250.0,
        step=0.5,
        help="Your height in centimeters."
    )
    smoking_freq = st.number_input(
        "🚬 Smoking Frequency (cigarettes per day)",
        min_value=0.0,
        max_value=50.0,
        step=1.0,
        help="Number of cigarettes you smoke per day (enter 0 if non-smoker)."
    )
    physical_hours = st.number_input(
        "🏃 Physical Activity (hours per week)",
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        help="Average hours you exercise per week."
    )
    alcohol_units = st.number_input(
        "🍷 Alcohol Consumption (units per week)",
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        help="Average alcohol units you consume per week."
    )
    family_history = st.selectbox(
        "🧬 Family History of Breast Cancer",
        options=["No", "Yes"],
        help="Select 'Yes' if you have a family history of breast cancer."
    )
    current_age = st.number_input(
        "📊 Current Age (years)",
        min_value=18,
        max_value=100,
        step=1,
        help="Your current age."
    )

# Compute BMI as a proxy for obesity rate
if height > 0:
    bmi = weight / ((height / 100) ** 2)
else:
    bmi = 0

# Map family history to binary
family_history_binary = 1 if family_history == "Yes" else 0

# Create input data array with 10 features:
# 1. Annual Income, 2. Annual Healthcare Spending, 3. Screening Years, 4. Access Rating,
# 5. BMI, 6. Smoking Frequency, 7. Physical Activity Hours, 8. Alcohol Units,
# 9. Family History (binary), 10. Current Age.
input_data = np.array([[
    annual_income,
    annual_healthcare,
    screening_years,
    access_rating,
    bmi,
    smoking_freq,
    physical_hours,
    alcohol_units,
    family_history_binary,
    current_age
]], dtype=np.float64)

# Check that the model expects 10 features
expected_features = getattr(selected_model, "n_features_in_", input_data.shape[1])
if input_data.shape[1] != expected_features:
    st.error(
        f"❌ Feature Mismatch: The selected model expects {expected_features} features, "
        f"but got {input_data.shape[1]}. Please retrain the model with the individual-level features."
    )
    st.stop()

if st.button("🔍 Predict Survival Rate"):
    try:
        prediction = selected_model.predict(input_data)[0]
        survival_percentage = max(0, min(100, prediction))
        st.success(f"🎯 Predicted Breast Cancer Survival Rate: **{survival_percentage:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
