from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from model import train_and_save_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "smart_manufacturing_data.csv"

WIDGET_KEYS = {
    "timestamp": "timestamp",
    "machine_id": "machine_id",
    "temperature": "temperature",
    "vibration": "vibration",
    "humidity": "humidity",
    "pressure": "pressure",
    "energy_consumption": "energy_consumption",
    "machine_status": "machine_status",
    "anomaly_flag": "anomaly_flag",
    "predicted_remaining_life": "predicted_remaining_life",
    "failure_type": "failure_type",
    "downtime_risk": "downtime_risk",
}


def get_presets() -> dict:
    current_time = datetime.now().replace(second=0, microsecond=0)
    return {
        "safe": {
            "timestamp": current_time,
            "machine_id": 1,
            "temperature": 68.0,
            "vibration": 24.0,
            "humidity": 40.0,
            "pressure": 3.2,
            "energy_consumption": 1.3,
            "machine_status": 1,
            "anomaly_flag": 0,
            "predicted_remaining_life": 420,
            "failure_type": "Normal",
            "downtime_risk": 0.0,
        },
        "high_risk": {
            "timestamp": current_time,
            "machine_id": 8,
            "temperature": 96.5,
            "vibration": 85.0,
            "humidity": 74.0,
            "pressure": 1.2,
            "energy_consumption": 4.8,
            "machine_status": 2,
            "anomaly_flag": 1,
            "predicted_remaining_life": 18,
            "failure_type": "Overheating",
            "downtime_risk": 1.0,
        },
    }


def set_preset_values(preset_name: str) -> None:
    presets = get_presets()
    preset = presets[preset_name]

    for key, value in preset.items():
        st.session_state[WIDGET_KEYS[key]] = value


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
    st.set_page_config(page_title="Smart Manufacturing", page_icon="⚙️", layout="wide")
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 1.25rem;
            background: linear-gradient(135deg, rgba(22, 97, 72, 0.95), rgba(26, 135, 84, 0.88));
            color: white;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
            margin-bottom: 1.25rem;
        }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p { margin: 0.35rem 0 0; opacity: 0.9; }
        .hint-box {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            background: rgba(255,255,255,0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Smart Manufacturing Maintenance Predictor</h1>
            <p>Use the preset values on the left to see a likely safe case or a likely maintenance case.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Quick Presets")
        st.write("Pick a preset to auto-fill the form.")
        if st.button("Load Safe Example", use_container_width=True):
            set_preset_values("safe")
            st.rerun()
        if st.button("Load High-Risk Example", use_container_width=True):
            set_preset_values("high_risk")
            st.rerun()

        st.divider()
        st.subheader("High-Risk Values")
        st.caption("These values are designed to increase the chance of maintenance_required = 1.")
        st.code(
            """temperature = 96.5
vibration = 85.0
humidity = 74.0
pressure = 1.2
energy_consumption = 4.8
machine_status = 2
anomaly_flag = 1
predicted_remaining_life = 18
failure_type = Overheating
downtime_risk = 1.0""",
            language="text",
        )

    model = ensure_model_ready()

    presets = get_presets()
    safe = presets["safe"]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            timestamp = st.datetime_input("Timestamp", value=safe["timestamp"], key=WIDGET_KEYS["timestamp"])
            machine_id = st.number_input("Machine ID", min_value=0, value=safe["machine_id"], step=1, key=WIDGET_KEYS["machine_id"])
            temperature = st.number_input("Temperature", value=safe["temperature"], key=WIDGET_KEYS["temperature"])
            vibration = st.number_input("Vibration", value=safe["vibration"], key=WIDGET_KEYS["vibration"])

        with c2:
            humidity = st.number_input("Humidity", value=safe["humidity"], key=WIDGET_KEYS["humidity"])
            pressure = st.number_input("Pressure", value=safe["pressure"], key=WIDGET_KEYS["pressure"])
            energy_consumption = st.number_input("Energy Consumption", value=safe["energy_consumption"], key=WIDGET_KEYS["energy_consumption"])
            machine_status = st.selectbox("Machine Status", options=[0, 1, 2], index=1, key=WIDGET_KEYS["machine_status"])

        with c3:
            anomaly_flag = st.selectbox("Anomaly Flag", options=[0, 1], index=0, key=WIDGET_KEYS["anomaly_flag"])
            predicted_remaining_life = st.number_input(
                "Predicted Remaining Life",
                min_value=0,
                value=safe["predicted_remaining_life"],
                step=1,
                key=WIDGET_KEYS["predicted_remaining_life"],
            )
            failure_type = st.selectbox(
                "Failure Type",
                options=["Normal", "Vibration Issue", "Overheating", "Pressure Drop"],
                index=0,
                key=WIDGET_KEYS["failure_type"],
            )
            downtime_risk = st.number_input(
                "Downtime Risk",
                min_value=0.0,
                max_value=1.0,
                value=safe["downtime_risk"],
                key=WIDGET_KEYS["downtime_risk"],
            )

        st.info("Tip: click Load High-Risk Example in the sidebar if you want to see a maintenance alert more often.")
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
            feature_view = pd.DataFrame([
                {
                    "machine_id": payload["machine_id"],
                    "temperature": payload["temperature"],
                    "vibration": payload["vibration"],
                    "humidity": payload["humidity"],
                    "pressure": payload["pressure"],
                    "energy_consumption": payload["energy_consumption"],
                    "machine_status": payload["machine_status"],
                    "anomaly_flag": payload["anomaly_flag"],
                    "predicted_remaining_life": payload["predicted_remaining_life"],
                    "failure_type": payload["failure_type"],
                    "downtime_risk": payload["downtime_risk"],
                    "hour": frame.iloc[0]["hour"],
                    "day_of_week": frame.iloc[0]["day_of_week"],
                }
            ])

            prediction = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0][1])

            result_label = "Maintenance Required" if prediction == 1 else "No Immediate Maintenance"
            result_color = st.error if prediction == 1 else st.success

            left, right = st.columns(2)
            with left:
                st.metric("Prediction", result_label)
            with right:
                st.metric("Maintenance Probability", f"{probability:.1%}")

            result_color(f"{result_label} (probability: {probability:.4f})")
            st.progress(int(probability * 100))

            st.subheader("Values Used")
            st.dataframe(feature_view, use_container_width=True)
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
