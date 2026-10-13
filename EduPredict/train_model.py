"""
EduPredict – Student Academic Performance Prediction System
============================================================
Script: train_model.py
Purpose:
  1. Load and validate the Kaggle Student Performance Dataset.
  2. Build reproducible scikit-learn preprocessing pipelines for numeric
     and categorical features.
  3. Train a Baseline Linear Regression model.
  4. Perform hyperparameter tuning on Random Forest using GridSearchCV (training data only).
  5. Evaluate both models on a held-out test set (MAE, RMSE, R²).
  6. Compare models, select the top performer, and persist the complete
     pipeline via Joblib for future deployment.
"""

import sys
from pathlib import Path

# ==============================================================================
# Configuration & Constants
# ==============================================================================
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "Student_Performance.csv"
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "best_model_pipeline.joblib"

NUMERIC_FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
CATEGORICAL_FEATURES = [
    "Extracurricular Activities",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Performance Index"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def check_dataset_exists(file_path: Path) -> None:
    """Validate that the dataset exists at the expected path."""
    if not file_path.exists():
        print("=" * 75)
        print(" [!] DATASET NOT FOUND")
        print(f" Expected location: {file_path.resolve()}")
        print("=" * 75)
        print("Follow these simple steps:")
        print("  1. Download the dataset from Kaggle:")
        print("     https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression")
        print("  2. Place 'Student_Performance.csv' directly into your 'data' folder:")
        print(f"     {file_path.parent.resolve()}\\Student_Performance.csv")
        print("  3. Run this script again: python train_model.py")
        print("=" * 75)
        sys.exit(1)


def check_dependencies():
    """Ensure required libraries are installed before importing."""
    missing_packages = []
    for pkg in ["numpy", "pandas", "sklearn", "joblib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append("scikit-learn" if pkg == "sklearn" else pkg)

    if missing_packages:
        print("=" * 75)
        print(" [!] MISSING DEPENDENCIES DETECTED")
        print(f" Missing packages: {', '.join(missing_packages)}")
        print("=" * 75)
        print("Please install required dependencies before running train_model.py:")
        print("  pip install -r requirements.txt")
        print("=" * 75)
        sys.exit(1)


def main():
    print("=" * 75)
    print(" EduPredict - Student Academic Performance Prediction System")
    print(" Stage 1: Model Training & Evaluation Pipeline")
    print("=" * 75)

    # Step 1: Pre-flight checks (dataset presence and dependencies)
    check_dataset_exists(DATA_FILE)
    check_dependencies()

    # Dynamic imports after pre-flight checks pass
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Step 2: Load and validate dataset
    print(f"\n[1/6] Loading dataset from: {DATA_FILE.resolve()}")
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    expected_columns = set(ALL_FEATURES + [TARGET_COLUMN])
    actual_columns = set(df.columns)
    missing_cols = expected_columns - actual_columns

    if missing_cols:
        print(f"Error: Dataset is missing required columns: {missing_cols}")
        print(f"Available columns in CSV: {list(df.columns)}")
        sys.exit(1)

    print(f" -> Successfully loaded {len(df):,} student records and {len(df.columns)} columns.")
    print(f" -> Input Features: {ALL_FEATURES}")
    print(f" -> Target Column:  {TARGET_COLUMN}")

    missing_count = df[ALL_FEATURES + [TARGET_COLUMN]].isnull().sum().sum()
    if missing_count > 0:
        print(f" -> Found {missing_count} missing value(s). Imputers will handle them during transformation.")
    else:
        print(" -> No missing values found. Imputer transformers are configured as safety guardrails.")

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # Step 3: Train / Test Split
    print(f"\n[2/6] Splitting dataset into train ({int((1-TEST_SIZE)*100)}%) and held-out test ({int(TEST_SIZE*100)}%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f" -> Training samples: {len(X_train):,}")
    print(f" -> Testing samples:  {len(X_test):,}")

    # Build preprocessing pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Step 4: Train Linear Regression (Baseline)
    print("\n[3/6] Training Linear Regression model pipeline...")
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    lr_pipeline.fit(X_train, y_train)
    print(" -> Linear Regression trained successfully.")

    # Step 5: Hyperparameter Tuning on Random Forest (Training Data Only)
    print("\n[4/6] Tuning Random Forest Regressor using GridSearchCV...")
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "regressor__n_estimators": [50, 100],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )

    print(" -> Running 5-fold cross-validation on training data only...")
    grid_search.fit(X_train, y_train)

    print(f" -> Best Cross-Validation R^2 Score: {grid_search.best_score_:.4f}")
    print(" -> Best Hyperparameters:")
    for param, val in grid_search.best_params_.items():
        print(f"    - {param}: {val}")

    best_rf_pipeline = grid_search.best_estimator_

    # Step 6: Evaluate both models on held-out test set
    print("\n[5/6] Evaluating models on held-out test set...")
    lr_pred = lr_pipeline.predict(X_test)
    rf_pred = best_rf_pipeline.predict(X_test)

    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    lr_metrics = compute_metrics(y_test, lr_pred)
    rf_metrics = compute_metrics(y_test, rf_pred)

    print("\n" + "=" * 70)
    print(f"{'MODEL EVALUATION RESULTS (HELD-OUT TEST SET)':^70}")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Linear Regression':<17} | {'Random Forest (Tuned)':<17}")
    print("-" * 70)
    print(f"{'Mean Absolute Error (MAE)':<30} | {lr_metrics['MAE']:<17.4f} | {rf_metrics['MAE']:<17.4f}")
    print(f"{'Root Mean Squared Error (RMSE)':<30} | {lr_metrics['RMSE']:<17.4f} | {rf_metrics['RMSE']:<17.4f}")
    print(f"{'R-squared (R^2 Score)':<30} | {lr_metrics['R2']:<17.4f} | {rf_metrics['R2']:<17.4f}")
    print("=" * 70)

    # Step 7: Select best model and save complete pipeline with Joblib
    print("\n[6/6] Selecting top model & exporting pipeline...")
    if rf_metrics["R2"] > lr_metrics["R2"]:
        best_name = "Random Forest Regressor (Tuned)"
        best_pipeline = best_rf_pipeline
        best_metrics = rf_metrics
    else:
        best_name = "Linear Regression"
        best_pipeline = lr_pipeline
        best_metrics = lr_metrics

    print(f" -> Top Model: {best_name}")
    print(f" -> Best Test Score: R^2 = {best_metrics['R2']:.4f}, RMSE = {best_metrics['RMSE']:.4f}, MAE = {best_metrics['MAE']:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_FILE)
    print(f" -> Exported pipeline successfully saved to: {MODEL_FILE.resolve()}")

    # Also save directly to model/trained_model.pkl for web application loading
    web_model_dir = Path("model")
    web_model_dir.mkdir(parents=True, exist_ok=True)
    web_model_file = web_model_dir / "trained_model.pkl"
    joblib.dump(best_pipeline, web_model_file)
    print(f" -> Exported pipeline also saved to: {web_model_file.resolve()}")
    print("\n[DONE] Model training and export completed successfully.")


if __name__ == "__main__":
    main()
