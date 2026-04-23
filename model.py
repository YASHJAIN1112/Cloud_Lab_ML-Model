import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "smart_manufacturing_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    # Convert timestamp into model-friendly numerical features.
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["hour"] = ts.dt.hour
    frame["day_of_week"] = ts.dt.dayofweek

    frame = frame.drop(columns=["timestamp"])
    return frame


def train_and_save_model() -> None:
    df = pd.read_csv(DATA_PATH)
    df = build_training_frame(df)

    target_column = "maintenance_required"
    feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns]
    y = df[target_column].astype(int)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "target": target_column,
        "features": feature_columns,
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "accuracy": accuracy,
        "report": report,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model trained and saved to: {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_and_save_model()
