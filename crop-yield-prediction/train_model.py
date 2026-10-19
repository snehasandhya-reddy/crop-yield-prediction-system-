from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "crop_yield_demo.csv"

def main():
    df = pd.read_csv(DATA)
    features = ["crop", "rainfall_mm", "temperature_c", "fertilizer_kg_ha", "soil_type", "irrigation"]
    target = "yield_tons_ha"
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["crop", "soil_type", "irrigation"]),
        ("numeric", "passthrough", ["rainfall_mm", "temperature_c", "fertilizer_kg_ha"]),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Evaluation on synthetic demo data")
    print(f"MAE : {mean_absolute_error(y_test, pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.3f}")
    print(f"R²  : {r2_score(y_test, pred):.3f}")
    print("\nReminder: replace synthetic demo data with a validated real-world dataset for meaningful use.")

if __name__ == "__main__":
    main()
