"""CodSoft Task 1: Titanic Survival Prediction.

Place Kaggle's Titanic train.csv in this folder before running.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path(__file__).resolve().parent / "train.csv"

def main():
    if not DATA_PATH.exists():
        print("Dataset not found.")
        print("Download train.csv from https://www.kaggle.com/competitions/titanic/data")
        print(f"Save it here: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if "Survived" not in df.columns:
        raise ValueError("The CSV must contain a 'Survived' target column.")

    # Keep useful, commonly available features; drop identifiers and high-cardinality text.
    feature_candidates = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    features = [col for col in feature_candidates if col in df.columns]
    if not features:
        raise ValueError("No expected feature columns were found in the CSV.")

    X = df[features].copy()
    y = df["Survived"]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== Titanic Survival Prediction ===")
    print(f"Features used: {', '.join(features)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    sample = X.iloc[[0]]
    print("\nExample passenger prediction:")
    print(f"Predicted survival (0 = No, 1 = Yes): {int(model.predict(sample)[0])}")

if __name__ == "__main__":
    main()
