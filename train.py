import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os

DATA_PATH = os.getenv("DATA_PATH", "mechanical_failure.csv")
MODEL_OUT = os.getenv("MODEL_PATH", "model.joblib")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required = {"sensor_1", "sensor_2", "sensor_3", "operating_temp", "failure"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {sorted(list(required))}")

X = df[["sensor_1", "sensor_2", "sensor_3", "operating_temp"]]
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")
joblib.dump(model, MODEL_OUT)
print(f"Saved model → {MODEL_OUT}")
