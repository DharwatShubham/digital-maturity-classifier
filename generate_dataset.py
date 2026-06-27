import pandas as pd
import random

data = []

for i in range(200):

    cloud = random.choice(["Yes", "No"])

    iot = random.randint(0, 300)

    erp = random.choice(["Yes", "No"])

    ai = random.choice(["Yes", "No"])

    automation = random.choice(
        ["Low", "Medium", "High", "Very High"]
    )

    if iot < 40:
        stage = "Digitization"

    elif iot < 90:
        stage = "Digitalization"

    elif iot < 150:
        stage = "Integration"

    else:
        stage = "Intelligent Automation"

    data.append([
        cloud,
        iot,
        erp,
        ai,
        automation,
        stage
    ])

df = pd.DataFrame(data, columns=[
    "Cloud_Usage",
    "IoT_Devices",
    "ERP_System",
    "AI_Usage",
    "Automation_Level",
    "Maturity_Stage"
])

df.to_csv("industry4_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())