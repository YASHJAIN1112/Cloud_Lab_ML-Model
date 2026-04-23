from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from model import train_and_save_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "smart_manufacturing_data.csv"


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        # Rebuild model.pkl in the current environment to avoid sklearn pickle mismatch.
        train_and_save_model()
        return joblib.load(MODEL_PATH)


def ensure_model_ready():
    model = load_model()

    try:
        # Validate pipeline compatibility with the current sklearn runtime.
        sample = pd.read_csv(DATA_PATH, nrows=1)
        sample = sample.drop(columns=["maintenance_required"], errors="ignore")
        sample_frame = prepare_input(sample.to_dict(orient="records")[0])
        model.predict(sample_frame)
        return model
    except Exception:
        train_and_save_model()
        return load_model()


def prepare_input(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame([payload])

    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["hour"] = ts.dt.hour
    frame["day_of_week"] = ts.dt.dayofweek

    return frame.drop(columns=["timestamp"])


def main() -> None:
    st.set_page_config(page_title="Smart Manufacturing Predictor", page_icon="⚙️", layout="wide")
    st.title("Smart Manufacturing Maintenance Predictor")
    st.caption("Predict whether maintenance is required using trained ML model from model.pkl")

    model = ensure_model_ready()

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            timestamp = st.datetime_input("Timestamp", value=datetime.now())
            machine_id = st.number_input("Machine ID", min_value=0, value=1, step=1)
            temperature = st.number_input("Temperature", value=75.0)
            vibration = st.number_input("Vibration", value=45.0)

        with c2:
            humidity = st.number_input("Humidity", value=55.0)
            pressure = st.number_input("Pressure", value=2.5)
            energy_consumption = st.number_input("Energy Consumption", value=2.0)
            machine_status = st.selectbox("Machine Status", options=[0, 1, 2], index=1)

        with c3:
            anomaly_flag = st.selectbox("Anomaly Flag", options=[0, 1], index=0)
            predicted_remaining_life = st.number_input(
                "Predicted Remaining Life",
                min_value=0,
                value=200,
                step=1,
            )
            failure_type = st.selectbox(
                "Failure Type",
                options=["Normal", "Vibration Issue", "Overheating", "Pressure Drop"],
            )
            downtime_risk = st.number_input("Downtime Risk", min_value=0.0, max_value=1.0, value=0.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": int(machine_id),
            "temperature": float(temperature),
            "vibration": float(vibration),
            "humidity": float(humidity),
            "pressure": float(pressure),
            "energy_consumption": float(energy_consumption),
            "machine_status": int(machine_status),
            "anomaly_flag": int(anomaly_flag),
            "predicted_remaining_life": int(predicted_remaining_life),
            "failure_type": failure_type,
            "downtime_risk": float(downtime_risk),
        }

        try:
            frame = prepare_input(payload)
            prediction = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0][1])

            if prediction == 1:
                st.error(f"Maintenance Required (probability: {probability:.4f})")
            else:
                st.success(f"No Immediate Maintenance (probability: {probability:.4f})")

            st.subheader("Input Payload")
            st.json(payload)
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
