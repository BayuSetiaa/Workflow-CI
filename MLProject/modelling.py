import os
import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))
mlflow.set_experiment("heart_failure_ci_experiment")

# Load data
df = pd.read_csv("heart_preprocessing/heart.csv")

# Preprocessing
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col])

df["Age_Cholesterol"] = df["Age"] * df["Cholesterol"]
df["BP_HeartRate"] = df["RestingBP"] * df["MaxHR"]

X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Autolog + train
mlflow.sklearn.autolog()
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi: {acc:.4f}")