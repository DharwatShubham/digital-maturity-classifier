from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ------------------------------------
# Load Dataset
# ------------------------------------
df = pd.read_csv("industry4_dataset.csv")

# ------------------------------------
# Encode Categorical Columns
# ------------------------------------
encoder = LabelEncoder()

df["Cloud_Usage"] = encoder.fit_transform(df["Cloud_Usage"])
df["ERP_System"] = encoder.fit_transform(df["ERP_System"])
df["AI_Usage"] = encoder.fit_transform(df["AI_Usage"])
df["Automation_Level"] = encoder.fit_transform(df["Automation_Level"])
df["Maturity_Stage"] = encoder.fit_transform(df["Maturity_Stage"])

# ------------------------------------
# Features and Target
# ------------------------------------
X = df.drop("Maturity_Stage", axis=1)
y = df["Maturity_Stage"]

# ------------------------------------
# Split Dataset
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------
# Compare Models
# ------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    cv_score = cross_val_score(model, X, y, cv=5)

    print(f"\n{name}")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Cross Validation Accuracy : {cv_score.mean()*100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy

# ------------------------------------
# Hyperparameter Tuning
# ------------------------------------
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

rf = RandomForestClassifier(random_state=42)

parameters = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=parameters,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Accuracy:")
print(f"{grid.best_score_*100:.2f}%")

# ------------------------------------
# Best Model
# ------------------------------------
best_model = grid.best_estimator_

# Test Accuracy
y_pred = best_model.predict(X_test)

print("\nFinal Test Accuracy:")
print(f"{accuracy_score(y_test, y_pred)*100:.2f}%")

# ------------------------------------
# Save Model
# ------------------------------------
joblib.dump(best_model, "best_model.pkl")

print("\nModel saved successfully!")

# ------------------------------------
# Classification Report
# ------------------------------------
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ------------------------------------
# Confusion Matrix
# ------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.show()

# ------------------------------------
# Feature Importance
# ------------------------------------
importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")

plt.xlabel("Importance Score")

plt.tight_layout()

plt.show()