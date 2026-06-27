import streamlit as st
import joblib

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("best_model.pkl")

# Mapping predicted labels to stage names
stages = {
    0: "Digitalization",
    1: "Digitization",
    2: "Integration",
    3: "Intelligent Automation"
}

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Digital Maturity Stage Classifier",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🤖 Digital Maturity Stage Classifier")
st.write("Predict an organization's Industry 4.0 Digital Maturity Stage")

# -----------------------------
# User Inputs
# -----------------------------
cloud = st.selectbox(
    "Cloud Usage",
    ["Yes", "No"]
)

iot = st.number_input(
    "Number of IoT Devices",
    min_value=0,
    max_value=500,
    value=100
)

erp = st.selectbox(
    "ERP System",
    ["Yes", "No"]
)

ai = st.selectbox(
    "AI Usage",
    ["Yes", "No"]
)

automation = st.selectbox(
    "Automation Level",
    ["Low", "Medium", "High", "Very High"]
)

# -----------------------------
# Encode Inputs
# -----------------------------
cloud = 1 if cloud == "Yes" else 0
erp = 1 if erp == "Yes" else 0
ai = 1 if ai == "Yes" else 0

automation_map = {
    "High": 0,
    "Low": 1,
    "Medium": 2,
    "Very High": 3
}

automation = automation_map[automation]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict"):

    input_data = [[cloud, iot, erp, ai, automation]]

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    confidence = max(probabilities[0]) * 100

    predicted_stage = stages[prediction[0]]

    st.success(f"✅ Predicted Stage: **{predicted_stage}**")
    st.info(f"🎯 Confidence: **{confidence:.2f}%**")

    # Show all probabilities
    st.subheader("Prediction Probabilities")

    for i, prob in enumerate(probabilities[0]):
        st.progress(float(prob))
        st.write(f"**{stages[i]}:** {prob*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Shubham Dharwat")