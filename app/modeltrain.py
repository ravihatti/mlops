import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
df = pd.read_csv("history_data_20_march2.csv", on_bad_lines="skip")
print("start")
# Convert 'created_on' column to datetime
df["created_on"] = pd.to_datetime(df["created_on"])

# Label encode categorical features
label_encoder = LabelEncoder()
df["node_lable_encoded"] = label_encoder.fit_transform(df["node_lable"])
df["instance_encoded"] = label_encoder.fit_transform(df["instance"].astype(str))
print("1")
# Normalize timestamp
scaler = MinMaxScaler() 
df["timestamp_scaled"] = scaler.fit_transform(df["created_on"].values.reshape(-1, 1))
print("2")
# Structure data into JSON format
structured_data = defaultdict(lambda: defaultdict(list))
for _, row in df.iterrows():
    node_label = row["node_lable"]
    instance = row["instance"]
    structured_data[node_label][instance].append(
        [row["created_on"], row["cpu_utilization_percent"]]
    )
print("1")
# Convert to standard dict
final_data = dict(structured_data)
train_data = []
for node_label, instances in final_data.items():
    for instance, values in instances.items():
        for i in range(len(values) - 1):  # Ensure we have a time series
            train_data.append([
                node_label, instance,
                values[i][0],  # Timestamp
                values[i][1],  # Previous CPU Utilization
                values[i+1][1]  # Next CPU Utilization (Target)
            ])
print("3")
# Convert list to DataFrame
train_df = pd.DataFrame(train_data, columns=["node_lable", "instance", "timestamp", "cpu_prev", "cpu_next"])
rows = []
for node_label, instances in final_data.items():
    for instance, records in instances.items():
        for timestamp, cpu_util in records:
            rows.append([node_label, instance, pd.to_datetime(timestamp), cpu_util])
df = pd.DataFrame(rows, columns=["node_label", "instance", "timestamp", "cpu_utilization"])
# 📌 Encode `node_label` & `instance`
node_encoder = LabelEncoder()
instance_encoder = LabelEncoder()
df["node_id"] = node_encoder.fit_transform(df["node_label"])
df["instance_id"] = instance_encoder.fit_transform(df["instance"])
# 📌 Convert `timestamp` to seconds since start
df["timestamp"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
scaler = MinMaxScaler()
df["cpu_utilization"] = scaler.fit_transform(df[["cpu_utilization"]])
print("4")
# 📌 Create Sequences for XGBoost (Past 5 Points → Predict Next)
time_steps = 5
X, y = [], []

for node_id in df["node_id"].unique():
    node_df = df[df["node_id"] == node_id]
    for i in range(len(node_df) - time_steps):
        X.append(node_df[["timestamp", "cpu_utilization", "instance_id"]].values[i:i+time_steps].flatten())  
        y.append(node_df["cpu_utilization"].values[i+time_steps])  # Next CPU value
print("Training..")
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train XGBoost Model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)
joblib.dump(model, "xgboost_model.pkl")
print("model dump")