import pandas as pd
import joblib

from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "student_features.csv"
MODEL_PATH = PROJECT_DIR / "models" / "best_student_performance_model.pkl"

# Load dataset and trained model
df = pd.read_csv(DATA_PATH, sep=None, engine="python")
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
feature_columns = model_data["features"]

print("=" * 45)
print(" STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 45)

print("\nDataset contains", len(df), "student records.")

while True:
    try:
        choice = input(
            "\nEnter student record number (0 to "
            + str(len(df) - 1)
            + "), or q to quit: "
        )

        if choice.lower() == "q":
            print("\nThank you for using the system!")
            break

        index = int(choice)

        if index < 0 or index >= len(df):
            print("Invalid record number. Try again.")
            continue

        # Select student record
        student = df.drop(columns=["G3"]).iloc[[index]]

        # Convert categorical features
        student = pd.get_dummies(student, drop_first=True)

        # Match training feature columns
        student = student.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Predict final grade
        prediction = model.predict(student)[0]

        actual_grade = df.iloc[index]["G3"]

        print("\n--- Prediction Result ---")
        print("Student record:", index)
        print("Predicted final grade:", round(prediction, 2))
        print("Actual final grade:", actual_grade)

        difference = abs(prediction - actual_grade)
        print("Prediction difference:", round(difference, 2))

        print("-------------------------")

    except ValueError:
        print("Please enter a valid number or q.")