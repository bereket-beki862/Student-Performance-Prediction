import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "student_features.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

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

# Define parameter combinations
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

# Create base model
rf = RandomForestRegressor(random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("Starting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Model Results:")
print("MAE:", round(mae, 3))
print("R2 Score:", round(r2, 3))

# Save best model
joblib.dump(
    {
        "model": best_model,
        "features": X.columns.tolist()
    },
    MODEL_DIR / "tuned_random_forest_model.pkl"
)

# Save results
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(
    RESULTS_DIR / "hyperparameter_tuning_results.csv",
    index=False
)

print("\nBest model saved successfully!")
print("Tuning results saved successfully!")