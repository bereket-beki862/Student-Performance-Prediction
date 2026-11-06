import pandas as pd
import joblib

from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "models"

# Load model comparison results
results = pd.read_csv(
    RESULTS_DIR / "final_model_comparison.csv"
)

print("Model Comparison:")
print(results.round(3).to_string(index=False))

# Find model with the lowest MAE
best_model_name = results.loc[
    results["MAE"].idxmin(), "Model"
]

print("\nBest Model:", best_model_name)

# Select the corresponding saved model
if best_model_name == "Random Forest":

    model_data = joblib.load(
        MODEL_DIR / "random_forest_model.pkl"
    )

elif best_model_name == "Tuned Random Forest":

    model_data = joblib.load(
        MODEL_DIR / "tuned_random_forest_model.pkl"
    )

else:
    print("\nBaseline model is the best model.")
    print("It is a simple benchmark, not a saved production model.")
    raise SystemExit

# Save the best model
joblib.dump(
    model_data,
    MODEL_DIR / "best_student_performance_model.pkl"
)

print("\nBest model saved successfully!")
print("File: models/best_student_performance_model.pkl")