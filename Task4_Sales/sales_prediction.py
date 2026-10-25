"""CodSoft Task 4: Sales Prediction using Linear Regression.

Place Advertising.csv in this folder. Expected columns: TV, Radio, Newspaper, Sales.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "Advertising.csv"
REQUIRED = ["TV", "Radio", "Newspaper", "Sales"]

def main():
    if not DATA_PATH.exists():
        print("Dataset not found.")
        print("Download it from https://www.kaggle.com/datasets/ashydv/advertising-dataset")
        print(f"Save the CSV as: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    # Remove a common unnamed index column if present.
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {REQUIRED}")

    df = df[REQUIRED].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 5:
        raise ValueError("The dataset needs at least five complete rows.")

    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("\n=== Sales Prediction ===")
    print(f"Rows used: {len(df)}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R² score: {r2_score(y_test, predictions):.3f}")
    print("\nLearned coefficients:")
    for feature, coefficient in zip(X.columns, model.coef_):
        print(f"  {feature}: {coefficient:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")

    sample = pd.DataFrame([{"TV": 100, "Radio": 20, "Newspaper": 10}])
    predicted_sales = model.predict(sample)[0]
    print("\nExample advertising spend (TV=100, Radio=20, Newspaper=10):")
    print(f"Predicted sales: {predicted_sales:.2f}")

if __name__ == "__main__":
    main()
