# KrishiSevak: An End-to-End AIoT Platform for Smart & Precision Agriculture

KrishiSevak is an enterprise-grade, edge-to-cloud Artificial Intelligence of Things (AIoT) platform designed to maximize crop yield and optimize field irrigation. The system operates in two strategic phases:
1. **Strategic Crop Recommendation:** Uses historical environment and soil parameters (NPK, pH, Rainfall, Climate) processed via an automated production MLOps pipeline to predict the most optimal crop to plant.
2. **Operational Smart Irrigation (Hardware Mesh):** Connects physical microcontrollers and capacitive moisture sensors deployed on real crop fields to stream real-time data back to a cloud backend, generating live alerts determining exactly *when* and *how much* to irrigate.

---

## 🏗 System Architecture & Data Flow

[ IoT Sensor Mesh ] --(HTTP/MQTT Stream)--> [ FastAPI Backend Engine ] <--> [ MongoDB Atlas ]
|
[ AWS S3 Model Registry ]
|
[ GitHub Actions CI/CD ] ---> [ AWS ECR (Docker) ] ---> [ AWS EC2 Live Deployment ]

1. **The Edge (Hardware Layer):** Ground-deployed capacitive moisture sensors capture Volumetric Water Content (VWC) percentages, streaming telemetry data directly to the cloud backend.
2. **The Ingestion & Processing Layer:** A modular **FastAPI** web framework routes sensor traffic, stores operational timelines inside **MongoDB**, and serves predictions using a productionized Random Forest classifier.
3. **The MLOps Lifecycle:** Fully automated data orchestration from validation to evaluation. Trained artifacts are securely pushed to **AWS S3** and continuous integration continuously packages code into **AWS ECR** containers deployed onto **AWS EC2**.

---

## 📂 Project Repository Directory Structure

```micro-layout
├── app.py                      # Main entrypoint exposing FastAPI endpoints & Web Interface
├── Dockerfile                  # Containerization template for consistent AWS cloud deployment
├── requirements.txt            # System dependencies and frozen package requirements
├── setup.py                    # Modular build script packaging local source logic as an installable egg
├── config/                     # Pipeline declarative configurations
│   ├── model.yaml              # Hyperparameters and model training configurations
│   └── schema.yaml             # Data validation schemas preventing downstream data drift
├── artifact/                   # Locally tracked, timestamped run outputs (Ingestion -> Trainer)
├── model_ml/                   # Production-ready serialized model and tracking objects
│   ├── label_encoder.joblib
│   └── random_forest_crop_model.joblib
├── notebook/                   # Research and development exploratory sandboxes
│   ├── crop_prediction.ipynb   # Model architecture exploration
│   └── mongo_db_demo.ipynb     # Database validation exploration
├── src/                        # Core Application Engine
│   ├── cloud_storage/          # AWS S3 abstract interface utilities
│   ├── components/             # Monolithic pipeline operational building blocks
│   │   ├── data_ingestion.py   # Raw data fetching from MongoDB to pipeline
│   │   ├── data_validation.py  # Data validation against schema.yaml constraints
│   │   ├── data_transformation.py # Preprocessing, scaling, and feature engineering
│   │   ├── model_trainer.py    # Model optimization and training loops
│   │   ├── model_evaluation.py # Validates production metrics against existing registry baselines
│   │   └── model_pusher.py     # Deploys champion model to production storage
│   ├── configuration/          # Remote Database & Cloud Engine clients
│   │   ├── aws_connection.py   # AWS Boto3 session hooks
│   │   └── mongo_db_connection.py # PyMongo connection clients
│   ├── entity/                 # Strongly typed custom Python data types and configurations
│   ├── pipline/                # Production loops running orchestration
│   │   ├── training_pipeline.py   # Complete automated end-to-end retraining run
│   │   └── prediction_pipeline.py # Real-time scoring and prediction requests
│   └── utils/                  # Shared helper scripts and serialization components
└── templates/                  # Frontend user portal components (Web Interface)

🛠 Tech Stack & Infrastructure Modules
Core Machine Learning: Python, Scikit-Learn, Pandas, NumPy, Joblib
API & Web Framework: FastAPI, Uvicorn, Jinja2 (HTML Templates)
Database Layer: MongoDB Atlas (NoSQL Document Store for ingestion & hardware state metadata)
Cloud Infrastructure (AWS): * EC2 (Elastic Compute Cloud): Hosts the production application server.
ECR (Elastic Container Registry): Secure storage registry for application Docker images.
S3 (Simple Storage Service): Secure production model registry and artifact cloud house.
DevOps / MLOps Pipeline: GitHub Actions (CI/CD Automated Deployment workflows), Docker.

🛰 Hardware & Smart Irrigation Integration Logic
The hardware layer implements a Dynamic Adaptive Threshold Control Loop. Soil requirements change depending on the crop choice confirmed by the machine learning algorithm.


[ ML Model Recommends Crop ] 
              │
              ▼
   [ Fetch Crop Water Tier ] ──► (e.g., Maize requires 40% Minimum VWC)
              │
              ▼
 [ Live Sensor Data Influx ] ──► (If Live VWC drops below 40% Threshold)
              │
              ▼
 [ Trigger Irrigation Alert ]

 Sensor Calibration: Raw ADC values (0−1023) are mapped into stable Volumetric Water Content percentages using calibration constants (ADC 
dry
​	
  vs ADC 
wet
​	
 ).
Data-Driven Smart Thresholding: The backend categorizes crop requirements into specific moisture tiers derived from statistical analysis of the historical data (Crop_recommendation.csv):
High Moisture Tiers (e.g., Rice, Jute): Alert triggers if live sensor drops below 70% VWC.
Moderate Moisture Tiers (e.g., Maize, Groundnuts): Alert triggers if live sensor drops below 40% VWC.
Arid Moisture Tiers (e.g., Watermelon, Muskmelon): Alert triggers if live sensor drops below 25% VWC.

🚀 Getting Started & Local Setup
Prerequisites
Python 3.8+ installed locally
A running MongoDB Atlas instance
AWS CLI configured with appropriate S3, ECR, and EC2 permissions
Local Installation
Clone the repository:

```bash
git clone [https://github.com/yourusername/krishisevak.git](https://github.com/yourusername/krishisevak.git)
cd krishisevak
```

```bash
git clone [https://github.com/yourusername/krishisevak.git](https://github.com/yourusername/krishisevak.git)
cd krishisevak
```

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

```bash
Bash
pip install -r requirements.txt
pip install -e .
```

```bash
MONGO_DB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
AWS_DEFAULT_REGION="us-east-1"
```

```bash
MONGO_DB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
AWS_DEFAULT_REGION="us-east-1"
```

```bash
python app.py
```

🔄 CI/CD Production Deployment (GitHub Actions)
This project includes fully automated workflow logic mapping deployment hooks straight to AWS production:
Continuous Integration (CI): On every push to the main branch, GitHub Actions builds and verifies unit tests, checks coding standards, and runs the validation module.
Continuous Deployment (CD): * Compiles the code using the project's Dockerfile.
Authenticates against AWS credentials and pushes the fresh container image into AWS ECR.
Pulls down the new container inside the target AWS EC2 server environment, cleanly hot-swapping the old container with zero downtime.