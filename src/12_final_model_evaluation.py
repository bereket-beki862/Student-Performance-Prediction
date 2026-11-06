import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "student_features.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Load dataset
df = pd.read_csv(DATA_PATH, sep=None, engine="python")

# Separate features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# Convert categorical features
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained models
# Load trained models
rf_data = joblib.load(MODEL_DIR / "random_forest_model.pkl")
tuned_data = joblib.load(MODEL_DIR / "tuned_random_forest_model.pkl")

models = {
    "Random Forest": rf_data["model"],
    "Tuned Random Forest": tuned_data["model"]
}
# Evaluate each model
results = []

for name, model in models.items():

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

# Create comparison table
comparison = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print("======================")
print(comparison.round(3).to_string(index=False))

# Save comparison
comparison.to_csv(
    RESULTS_DIR / "final_model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")