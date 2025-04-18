from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("xgboost_model.pkl")

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}

def save_actual_vs_prediction(output_file="cpu_predictions.csv"):
    """Save actual and predicted CPU utilization for all devices-instances to an Excel file."""
    results = []

    for node_label in node_encoder.classes_:
        for instance in instance_encoder.classes_:
            node_id = node_encoder.transform([node_label])[0]
            instance_id = instance_encoder.transform([instance])[0]

            node_df = df[(df["node_id"] == node_id) & (df["instance_id"] == instance_id)]

            if len(node_df) < time_steps:
                continue  # Skip if not enough data

            for i in range(len(node_df) - time_steps):
                past_data = node_df[["timestamp", "cpu_utilization", "instance_id"]].values[i:i+time_steps].flatten().reshape(1, -1)
                prediction = model.predict(past_data)

                predicted_value = scaler.inverse_transform([[prediction[0]]])[0][0]
                actual_value = scaler.inverse_transform([[node_df["cpu_utilization"].values[i+time_steps]]])[0][0]
                timestamp = pd.to_datetime(df["timestamp"].min()) + pd.to_timedelta(node_df["timestamp"].values[i+time_steps], unit="s")

                results.append([node_label, instance, timestamp, actual_value, predicted_value])

    # 📌 Save to Excel
    result_df = pd.DataFrame(results, columns=["Node Label", "Instance", "Timestamp", "Actual CPU Utilization", "Predicted CPU Utilization"])
    result_df.to_csv(output_file, index=False)
    print(f"✅ Data saved to {output_file}")
    return {"prediction":output_file}
