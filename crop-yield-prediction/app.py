from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_yield_demo.csv"

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    data = load_data()
    features = ["crop", "rainfall_mm", "temperature_c", "fertilizer_kg_ha", "soil_type", "irrigation"]
    target = "yield_tons_ha"
    X = data[features]
    y = data[target]
    categorical = ["crop", "soil_type", "irrigation"]
    numeric = ["rainfall_mm", "temperature_c", "fertilizer_kg_ha"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics, data

st.title("🌾 Crop Yield Prediction System")
st.caption("A machine-learning demonstration that estimates crop yield from selected agricultural inputs.")
st.warning(
    "Demo-data notice: this version uses synthetically generated example data to demonstrate the workflow. "
    "Predictions are not suitable for real farming decisions. Replace the demo dataset with a validated "
    "regional agricultural dataset before making real-world claims."
)

model, metrics, data = train_model()
tab_predict, tab_explore, tab_about = st.tabs(["🔮 Predict yield", "📊 Explore data", "ℹ️ About"])

with tab_predict:
    st.subheader("Enter agricultural conditions")
    left, right = st.columns(2)
    with left:
        crop = st.selectbox("Crop", sorted(data["crop"].unique()))
        rainfall = st.slider("Seasonal rainfall (mm)", 100, 2500, 1000, 50)
        temperature = st.slider("Average temperature (°C)", 10.0, 42.0, 26.0, 0.5)
    with right:
        fertilizer = st.slider("Fertilizer used (kg/ha)", 0, 300, 100, 5)
        soil = st.selectbox("Soil type", sorted(data["soil_type"].unique()))
        irrigation = st.selectbox("Irrigation available?", ["Yes", "No"])
    if st.button("Predict crop yield", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "crop": crop,
            "rainfall_mm": rainfall,
            "temperature_c": temperature,
            "fertilizer_kg_ha": fertilizer,
            "soil_type": soil,
            "irrigation": irrigation,
        }])
        prediction = float(model.predict(row)[0])
        st.success(f"Estimated yield: **{prediction:.2f} tons/hectare**")
        st.info("This is a model estimate from demo data, not an agricultural recommendation.")

with tab_explore:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Demo records", f"{len(data):,}")
    b.metric("Crop categories", data["crop"].nunique())
    c.metric("Input features", 6)
    st.dataframe(data.head(20), use_container_width=True)
    st.subheader("Average yield by crop (demo data)")
    avg = data.groupby("crop", as_index=False)["yield_tons_ha"].mean().sort_values("yield_tons_ha")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(avg["crop"], avg["yield_tons_ha"])
    ax.set_xlabel("Average yield (tons/hectare)")
    ax.set_ylabel("Crop")
    ax.set_title("Average yield by crop — synthetic demo data")
    fig.tight_layout()
    st.pyplot(fig)
    st.subheader("Input relationships")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.scatter(data["rainfall_mm"], data["yield_tons_ha"], alpha=0.45)
    ax2.set_xlabel("Rainfall (mm)")
    ax2.set_ylabel("Yield (tons/hectare)")
    ax2.set_title("Rainfall vs yield — synthetic demo data")
    fig2.tight_layout()
    st.pyplot(fig2)

with tab_about:
    st.subheader("Project workflow")
    st.markdown("""
    1. Load agricultural records from a CSV file.
    2. Separate input features and the target yield.
    3. Encode categorical variables and pass through numeric variables.
    4. Train a Random Forest regression model.
    5. Evaluate the model on a held-out test split.
    6. Use the trained pipeline to estimate yield for user-entered conditions.
    """)
    st.subheader("Model evaluation on held-out demo data")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['mae']:.3f}")
    m2.metric("RMSE", f"{metrics['rmse']:.3f}")
    m3.metric("R²", f"{metrics['r2']:.3f}")
    st.caption("Metrics describe performance on this synthetic demonstration dataset only.")
    st.subheader("Feature definitions")
    st.markdown("""
    - **Rainfall:** seasonal rainfall in millimetres.
    - **Temperature:** average temperature in degrees Celsius.
    - **Fertilizer:** illustrative quantity in kilograms per hectare.
    - **Soil type:** categorical soil label used by the demo dataset.
    - **Irrigation:** whether irrigation is available.
    - **Yield:** target value expressed in tons per hectare.
    """)
