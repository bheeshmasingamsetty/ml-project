from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

model = None
scaler = None
feature_columns = None
dataset = None
metrics = {}
last_predictions = None

REQUIRED = [
    "ENGINESIZE", "CYLINDERS", "TRANSMISSION", "FUELTYPE",
    "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY",
    "FUELCONSUMPTION_COMB", "CO2EMISSIONS"
]

def prepare_dataframe(df):
    df = df.copy()
    # Same cleaning direction used in the notebook.
    for col in ["MAKE", "MODEL"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.dropna().drop_duplicates()
    return df

def train_model(df):
    global model, scaler, feature_columns, metrics, last_predictions

    df = prepare_dataframe(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    X = df[[
        "ENGINESIZE", "CYLINDERS", "TRANSMISSION", "FUELTYPE",
        "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB"
    ]]
    y = df["CO2EMISSIONS"]

    X = pd.get_dummies(X, columns=["FUELTYPE", "TRANSMISSION"], drop_first=True)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": round(float(mae), 2),
        "mse": round(float(mse), 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "rows": int(len(df))
    }

    last_predictions = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": y_pred
    }).sort_index()

    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload():
    global dataset
    if "file" not in request.files:
        return jsonify({"error": "Please select a CSV file."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400

    try:
        raw = pd.read_csv(file)
        dataset = train_model(raw)
        preview = dataset.head(8).replace({np.nan: None}).to_dict(orient="records")

        numeric = dataset.select_dtypes(include=np.number)
        corr = numeric.corr(numeric_only=True).round(3).fillna(0)

        return jsonify({
            "message": "Dataset loaded and Elastic Net model trained successfully.",
            "preview": preview,
            "columns": dataset.columns.tolist(),
            "metrics": metrics,
            "correlation": {
                "columns": corr.columns.tolist(),
                "values": corr.values.tolist()
            },
            "summary": {
                "rows": int(dataset.shape[0]),
                "columns": int(dataset.shape[1]),
                "mean_co2": round(float(dataset["CO2EMISSIONS"].mean()), 2),
                "min_co2": round(float(dataset["CO2EMISSIONS"].min()), 2),
                "max_co2": round(float(dataset["CO2EMISSIONS"].max()), 2)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or dataset is None:
        return jsonify({"error": "Upload a CSV and train the model first."}), 400

    try:
        data = request.get_json(force=True)

        row = {
            "ENGINESIZE": float(data["engine"]),
            "CYLINDERS": float(data["cylinders"]),
            "TRANSMISSION": str(data["transmission"]),
            "FUELTYPE": str(data["fueltype"]),
            "FUELCONSUMPTION_CITY": float(data["city"]),
            "FUELCONSUMPTION_HWY": float(data["highway"]),
            "FUELCONSUMPTION_COMB": float(data["combined"])
        }

        X = pd.DataFrame([row])
        X = pd.get_dummies(X, columns=["FUELTYPE", "TRANSMISSION"], drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(X)
        prediction = float(model.predict(X_scaled)[0])

        return jsonify({
            "prediction": round(prediction, 2),
            "unit": "g/km"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predictions")
def predictions():
    if last_predictions is None:
        return jsonify({"actual": [], "predicted": []})
    return jsonify({
        "actual": [round(float(x), 2) for x in last_predictions["actual"]],
        "predicted": [round(float(x), 2) for x in last_predictions["predicted"]]
    })

if __name__ == "__main__":
    app.run(debug=True)
