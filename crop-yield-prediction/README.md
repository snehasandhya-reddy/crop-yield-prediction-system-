# 🌾 Crop Yield Prediction System Using Machine Learning

A small end-to-end Python/ML project that demonstrates crop-yield regression through a Streamlit web app.

## Features
- User inputs for crop, rainfall, temperature, fertilizer use, soil type, and irrigation.
- Random Forest regression pipeline with categorical encoding.
- Evaluation metrics: MAE, RMSE, and R².
- Exploratory dataset table and visualizations.
- Separate training/evaluation script.

## Important dataset notice
The included `data/crop_yield_demo.csv` is **synthetically generated demonstration data**, not a verified agricultural dataset. It exists so the project can run immediately and demonstrate the full workflow. Its predictions and metrics must **not** be used for actual farming decisions or represented as validated agricultural performance.

For a research-quality version, replace it with a credible, documented agricultural dataset for the intended region and crop. Check the dataset's license, units, collection period, and target definition. Update preprocessing/features to match the new data and re-evaluate the model.

## Requirements
- Python 3.10 or newer recommended
- pip

## Run locally (Windows / VS Code)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install packages
python -m pip install -r requirements.txt

# 3. Run the web application
streamlit run app.py
```

A browser tab should open with the app. If it does not, open the local URL printed in the terminal.

## Evaluate the model from the terminal

```bash
python train_model.py
```

The script reports MAE, RMSE, and R² on a held-out test split. Since the bundled data is synthetic, these metrics are only a software demonstration.

## Dataset columns

| Column | Meaning | Type |
|---|---|---|
| `crop` | Crop category | Categorical |
| `rainfall_mm` | Seasonal rainfall in mm | Numeric |
| `temperature_c` | Average temperature in °C | Numeric |
| `fertilizer_kg_ha` | Fertilizer quantity in kg/ha | Numeric |
| `soil_type` | Soil category | Categorical |
| `irrigation` | Irrigation available (Yes/No) | Categorical |
| `yield_tons_ha` | Target yield in tons/hectare | Numeric |

## Workflow
1. Load CSV data using Pandas.
2. Split features and target.
3. Encode categorical features with `OneHotEncoder`.
4. Train a `RandomForestRegressor` using a scikit-learn pipeline.
5. Evaluate on a held-out test set.
6. Collect user inputs in Streamlit and return a yield estimate.

## Limitations and future improvements
- Replace the synthetic demo data with validated real-world data.
- Add region, season, soil nutrients, crop variety, and historical yield where available.
- Use cross-validation and compare multiple regression algorithms.
- Add data validation, model versioning, tests, and model monitoring.
- Do not treat model output as professional agronomic advice.

## Suggested GitHub description
`An end-to-end machine learning demo for crop-yield estimation with a Random Forest pipeline and Streamlit dashboard.`

## License
Add a license only after deciding how you want others to reuse your code. Verify the license of any external dataset you add.
