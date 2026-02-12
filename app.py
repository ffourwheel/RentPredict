from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
import os
import uuid

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CSV_FILE = os.path.join(DATA_DIR, "default_data.csv")
DATA_FILE = os.path.join(DATA_DIR, "data.json")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_info.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, encoding="utf-8")
        data = []
        for _, row in df.iterrows():
            data.append({
                "id": str(uuid.uuid4()),
                "name": str(row["name"]),
                "distance": float(row["distance"]),
                "room_size": float(row["room_size"]),
                "convenience": int(row["convenience"]),
                "fitness": int(row["fitness"]),
                "room_condition": int(row["room_condition"]),
                "price": float(row["price"]),
            })
        save_data(data)
        return data
    return []


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data", methods=["GET"])
def get_data():
    return jsonify(load_data())


@app.route("/api/data", methods=["POST"])
def add_data():
    entry = request.json
    entry["id"] = str(uuid.uuid4())
    required = ["name", "distance", "room_size", "convenience", "fitness", "room_condition", "price"]
    for field in required:
        if field not in entry:
            return jsonify({"error": f"Missing field: {field}"}), 400
    data = load_data()
    data.append(entry)
    save_data(data)
    return jsonify(entry), 201


@app.route("/api/data/<entry_id>", methods=["DELETE"])
def delete_data(entry_id):
    data = load_data()
    data = [d for d in data if d["id"] != entry_id]
    save_data(data)
    return jsonify({"success": True})


@app.route("/api/train", methods=["POST"])
def train_model():
    data = load_data()
    if len(data) < 5:
        return jsonify({"error": "ต้องมีข้อมูลอย่างน้อย 5 รายการ"}), 400

    df = pd.DataFrame(data)
    feature_cols = ["distance", "room_size", "convenience", "fitness", "room_condition"]
    X = df[feature_cols].values.astype(float)
    y = df["price"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred) if len(y_test) > 1 else float("nan")
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    joblib.dump(model, MODEL_FILE)

    coefficients = {}
    for i, col in enumerate(feature_cols):
        coefficients[col] = round(float(model.coef_[i]), 4)

    model_info = {
        "trained": True,
        "intercept": round(float(model.intercept_), 4),
        "coefficients": coefficients,
        "train_r2": round(float(train_r2), 4),
        "test_r2": round(float(test_r2), 4) if not np.isnan(test_r2) else "N/A",
        "test_mae": round(float(test_mae), 2),
        "test_mse": round(float(test_mse), 2),
        "test_rmse": round(float(test_rmse), 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_data": len(data),
        "train_actual": y_train.tolist(),
        "train_pred": [round(v, 2) for v in y_train_pred.tolist()],
        "test_actual": y_test.tolist(),
        "test_pred": [round(v, 2) for v in y_test_pred.tolist()],
        "feature_names": feature_cols,
    }

    with open(MODEL_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    return jsonify(model_info)


@app.route("/api/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "ยังไม่ได้ train model กรุณา train ก่อน"}), 400

    model = joblib.load(MODEL_FILE)
    body = request.json
    feature_cols = ["distance", "room_size", "convenience", "fitness", "room_condition"]
    try:
        features = [float(body[col]) for col in feature_cols]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    prediction = model.predict([features])[0]
    prediction = max(0, round(float(prediction), 2))
    return jsonify({"predicted_price": prediction, "features": body})


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    if os.path.exists(MODEL_INFO_FILE):
        with open(MODEL_INFO_FILE, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"trained": False})


if __name__ == "__main__":
    print("เปิดเบราว์เซอร์ไปที่: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
