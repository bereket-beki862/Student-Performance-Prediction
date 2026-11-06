# Student Performance Prediction

A Machine Learning project that analyzes student academic data and predicts students' final grades using regression models.

## Project Overview

This project uses the UCI Student Performance Dataset to explore factors associated with academic achievement and predict students' final grades.

The project covers data inspection, exploratory data analysis, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and prediction.

## Dataset

- **Source:** UCI Student Performance Dataset
- **Student records:** 395
- **Input features:** 35
- **Target variable:** `G3` (final grade)
- **Average final grade:** 10.42

The dataset contains information about students' academic performance, study habits, family background, and other relevant factors.

## Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   ├── student-mat.csv
│   └── student_features.csv
│
├── models/
│   ├── best_student_performance_model.pkl
│   ├── random_forest_model.pkl
│   └── tuned_random_forest_model.pkl
│
├── notebooks/
├── results/
├── src/
│   ├── 01_load_and_inspect.py
│   ├── 02_eda.py
│   ├── 03_data_preprocessing.py
│   ├── 04_feature_engineering.py
│   └── ...
│
├── .gitignore
└── README.md
```

## Technologies Used

- **Python** – Main programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning and model evaluation
- **Joblib** – Saving and loading trained models
- **Git and GitHub** – Version control and project hosting

## Machine Learning Models

The project evaluates Random Forest regression and an optimized Random Forest model using hyperparameter tuning.

### Model Performance

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Random Forest | 1.138 | 3.603 | 1.898 | 0.824 |
| Tuned Random Forest | **1.132** | **3.457** | **1.859** | **0.831** |

### Best Model: Tuned Random Forest

- **MAE:** 1.132
- **MSE:** 3.457
- **RMSE:** 1.859
- **R² Score:** 0.831

The tuned Random Forest achieved the best performance among the evaluated models.

An R² score of 0.831 indicates that the model explains approximately 83.1% of the variation in final grades on the evaluation data.

## Project Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Clean and preprocess the data.
4. Create additional features.
5. Train a baseline regression model.
6. Train and compare Random Forest models.
7. Perform cross-validation.
8. Analyze feature importance.
9. Tune model hyperparameters.
10. Evaluate the final model.
11. Save the best-performing model.
12. Generate predictions.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/bereket-beki862/Student-Performance-Prediction.git
```

### 2. Open the project directory

```bash
cd Student-Performance-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 5. Run the project

Run the Python scripts from the project root directory in their numerical order.

For example:

```bash
python src/01_load_and_inspect.py
python src/02_eda.py
python src/03_data_preprocessing.py
python src/04_feature_engineering.py
```

Continue with the remaining scripts in `src/` to train, evaluate, and save the models.

To use the prediction interface after preparing the trained model:

```bash
python src/14_prediction_interface.py
```

## Results

The project produces:

- Exploratory data analysis visualizations.
- Feature importance analysis.
- Model comparison results.
- Cross-validation and hyperparameter tuning results.
- Saved trained machine learning models.
- Student final-grade predictions.

Generated outputs are stored in the `results/` and `models/` directories.

## Future Improvements

- Explore additional regression algorithms.
- Improve model generalization through further experimentation.
- Develop a web-based interface for student performance prediction.
- Expand the analysis using additional student datasets.

## Author

**Bereket Kefyalew**

B.Tech Computer Science and Engineering

GitHub: [bereket-beki862](https://github.com/bereket-beki862)

---

*This project was developed as part of my practical learning journey in Python, Data Science, and Machine Learning.*