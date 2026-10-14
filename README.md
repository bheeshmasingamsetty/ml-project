# Vehicle CO₂ Emission Prediction Website

A Flask website based on the uploaded `ML_2.ipynb` notebook.

## What it does
- Uploads `FuelConsumption.csv`
- Cleans the dataset by dropping MAKE/MODEL, null values and duplicates
- One-hot encodes FUELTYPE and TRANSMISSION
- Splits data 80/20 with random_state=42
- Standardizes features with StandardScaler
- Trains ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
- Shows MAE, MSE, RMSE and R²
- Provides a vehicle CO₂ prediction form
- Shows actual-vs-predicted, engine-size and fuel-consumption charts

## Run on Windows / VS Code

1. Install Python 3.10+.
2. Open this folder in VS Code.
3. Open Terminal.
4. Run:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

5. Install packages:

```bash
pip install -r requirements.txt
```

6. Start:

```bash
python app.py
```

7. Open the address shown in the terminal, normally:
`http://127.0.0.1:5000`

8. Upload `FuelConsumption.csv`.

## Important
The notebook contains some intermediate/repeated cells and an upload error, so this website uses the clean final modeling workflow found in the notebook rather than reproducing the Colab upload error.
