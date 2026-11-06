import pandas as pd
import joblib

from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "student_features.csv"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "models"

# Load dataset and results
df = pd.read_csv(DATA_PATH, sep=None, engine="python")
results = pd.read_csv(
    RESULTS_DIR / "final_model_comparison.csv"
)

# Find best model using MAE
best_row = results.loc[results["MAE"].idxmin()]

best_model_name = best_row["Model"]
best_mae = best_row["MAE"]
best_rmse = best_row["RMSE"]
best_r2 = best_row["R2 Score"]

# Project information
total_students = len(df)
total_features = len(df.columns) - 1
average_grade = df["G3"].mean()

# Generate summary
summary = f"""
STUDENT PERFORMANCE PREDICTION
================================

PROJECT OVERVIEW

Project: Student Performance Prediction
Dataset: UCI Student Performance Dataset

DATASET INFORMATION

Total student records: {total_students}
Number of input features: {total_features}
Target variable: G3
Average final grade: {average_grade:.2f}

MODEL PERFORMANCE

{results.round(3).to_string(index=False)}

BEST MODEL

Model: {best_model_name}
MAE: {best_mae:.3f}
RMSE: {best_rmse:.3f}
R2 Score: {best_r2:.3f}

PROJECT CONCLUSION

This project uses machine learning regression models
to predict students' final academic performance.

The models were evaluated using MAE, RMSE, and R2 Score.

The best-performing model was selected based on
the lowest Mean Absolute Error.

The project demonstrates data preprocessing,
feature engineering, model training, evaluation,
and prediction.

================================
END OF PROJECT SUMMARY
"""

# Save summary
summary_path = RESULTS_DIR / "project_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write(summary)

print(summary)
print("\nProject summary saved successfully!")
print("Location:", summary_path)