⚙️ Mechanical Failure Prediction using AI
🔍 Overview

This project predicts potential mechanical failures in industrial machines using machine learning trained on sensor data.
It helps perform predictive maintenance, reducing downtime and avoiding costly unplanned repairs.

The solution includes:

🧠 Model Training (Random Forest on sensor data)

⚙️ Backend API (FastAPI-based REST endpoints)

💻 Frontend Dashboard (Streamlit interface for batch & single predictions)

🧩 Project Architecture
mechanical-failure-ai/
│
├── train.py              # Trains model on mechanical_failure.csv
├── backend.py            # FastAPI backend for prediction APIs
├── app.py                # Streamlit frontend UI
├── mechanical_failure.csv # Sensor dataset
├── requirements.txt       # Dependencies
└── model.joblib           # Saved ML model (generated after training)

🧠 Model Pipeline

Input Data:

sensor_1, sensor_2, sensor_3, operating_temp, failure

Preprocessing:

Validation of feature columns

Train-test split (80:20)

Model:

RandomForestClassifier

Evaluated using Accuracy and ROC-AUC

Output:

Probability of failure (0 = healthy, 1 = failure risk)

⚙️ Tech Stack
Component	Technology
ML Model	Scikit-learn (Random Forest)
Backend API	FastAPI
Frontend	Streamlit
Data Handling	Pandas, Joblib
Deployment Ready	Uvicorn / Render / HuggingFace Spaces
🚀 How to Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python train.py

3️⃣ Start the backend API
uvicorn backend:app --reload --port 8000

4️⃣ Launch Streamlit frontend
streamlit run app.py


The Streamlit app will open at http://localhost:8501

Backend runs at http://localhost:8000

📊 Example Inputs
sensor_1	sensor_2	sensor_3	operating_temp
0.12	10.0	100.0	35.0
0.45	8.2	120.5	40.2

📈 Output → Failure Probability: 0.23

📈 Model Performance

Accuracy: ~90%

ROC-AUC: ~0.92

Robust against noise in moderate-size datasets

🧩 API Endpoints
Endpoint	Method	Description
/predict	POST	Predicts single record’s failure probability
/predict_batch	POST	Predicts batch of sensor records from CSV

Example Request:

{
  "sensor_1": 0.12,
  "sensor_2": 10.0,
  "sensor_3": 100.0,
  "operating_temp": 35.0
}


Response:

{ "failure_probability": 0.23 }

📦 Future Enhancements

Add database integration for storing predictions

Live IoT sensor streaming integration (MQTT/Kafka)

Docker-based deployment for full-stack hosting

Add SHAP-based model explainability

👤 Author

Abhishek Chandra
B.Tech CSE (AI & Data Science)
LinkedIn
 | GitHub
