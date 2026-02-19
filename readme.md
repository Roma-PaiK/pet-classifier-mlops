# Pet Classifier MLOps Project 🐾

An end-to-end MLOps pipeline for classifying images of Cats and Dogs. This project demonstrates model development, experiment tracking, data versioning, containerization, and automated CI/CD deployment.

---

## 📁 Project Structure

```text
pet-classifier-mlops/
├── .github/workflows/      # (Upcoming) CI/CD Pipeline definitions
├── data/                   # Raw and processed datasets (Tracked via DVC)
├── models/                 # Serialized trained models (.h5) (Tracked via DVC)
├── notebooks/              # Jupyter notebooks for model training and EDA
├── mlruns/                 # Local MLflow experiment tracking logs
├── app.py                  # FastAPI inference service
├── Dockerfile              # Containerization blueprint
├── requirements.txt        # Pinned dependencies for the production environment
├── .dvc/                   # DVC pipeline and tracking config
├── .gitignore              # Git ignore rules for heavy files
└── README.md               # Project documentation
```

---

## ✅ Completed Milestones

### M1: Model Development & Experiment Tracking

- Built a baseline Convolutional Neural Network (CNN) using TensorFlow 2.19.0  
- Versioned the raw dataset (PetImages) and serialized model (`baseline_cnn.h5`) using DVC  
- Tracked experiments, hyperparameters, and artifacts using MLflow  

### M2: Model Packaging & Containerization

- Packaged the trained model into a REST API using FastAPI  
- Implemented `/health` and `/predict` endpoints  
- Pinned strict environment dependencies in `requirements.txt`  
- Containerized the inference service using Docker  

---

## 🛠️ How to Run Locally

### 1. Prerequisites

Ensure the following tools are installed:

- Git  
- DVC  
- Docker Desktop  
- Python 3.10+  

---

### 2. Clone and Setup

Clone the repository and pull the heavy files (data and models) from the DVC remote storage.

```bash
git clone https://github.com/YourUsername/pet-classifier-mlops.git
cd pet-classifier-mlops

# Pull dataset and model files tracked by DVC
dvc pull
```

---

### 3. Run the Containerized API

Build the Docker image and start the FastAPI server on port 8000:

```bash
docker build -t pet-classifier-api .
docker run -p 8000:8000 pet-classifier-api
```

---

### 4. Test the Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Prediction

Option 1: Open Swagger UI in your browser:

```
http://localhost:8000/docs
```

Option 2: Use cURL

```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

---
