import joblib
import pandas as pd

# Load trained model
model = joblib.load("best_model.pkl")

print("Digital Maturity Stage Predictor")
print("-" * 35)

# User Input
cloud = input("Cloud Usage (Yes/No): ")
iot = int(input("Number of IoT Devices: "))
erp = input("ERP System (Yes/No): ")
ai = input("AI Usage (Yes/No): ")
automation = input("Automation Level (Low/Medium/High/Very High): ")

# Encode manually
cloud = 1 if cloud.lower() == "yes" else 0
erp = 1 if erp.lower() == "yes" else 0
ai = 1 if ai.lower() == "yes" else 0

automation_map = {
    "High": 0,
    "Low": 1,
    "Medium": 2,
    "Very High": 3
}

automation = automation_map[automation]

# Create DataFrame
sample = pd.DataFrame([[
    cloud,
    iot,
    erp,
    ai,
    automation
]], columns=[
    "Cloud_Usage",
    "IoT_Devices",
    "ERP_System",
    "AI_Usage",
    "Automation_Level"
])

# Predict
prediction = model.predict(sample)[0]

# Decode Prediction
stages = {
    0: "Digitalization",
    1: "Digitization",
    2: "Integration",
    3: "Intelligent Automation"
}

print("\nPredicted Digital Maturity Stage:")
print(stages[prediction])
