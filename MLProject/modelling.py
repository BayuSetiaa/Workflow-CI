import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Set MLflow tracking URI & experiment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))
mlflow.set_experiment("heart_failure_ci_experiment")

# Load dataset
df = pd.read_csv("heart_preprocessing/heart.csv")

# Preprocessing
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col])

# Feature engineering
df["Age_Cholesterol"] = df["Age"] * df["Cholesterol"]
df["BP_HeartRate"] = df["RestingBP"] * df["MaxHR"]

X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

# Split & Resample
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Mulai run MLflow
with mlflow.start_run(run_name="model_training_with_registry") as run:
    mlflow.sklearn.autolog()

    # Train model
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")

    # Log & Register model ke Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="heart_failure_model"
    )

# Simpan model dan data uji lokal untuk monitoring
joblib.dump(model, "model.joblib")
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)