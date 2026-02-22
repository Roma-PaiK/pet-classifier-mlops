# Pet Classifier MLOps Project 🐾

An end-to-end MLOps pipeline for classifying images of Cats and Dogs. This project demonstrates model development, experiment tracking, data versioning, containerization, and automated CI/CD deployment.

---

## 📁 Project Structure

```text
pet-classifier-mlops/
├── .github/workflows/      # CI Pipeline definitions (M3)
├── data/                   # Raw and processed datasets (Tracked via DVC)
├── models/                 # Serialized trained models (.h5) (Tracked via DVC)
├── notebooks/              # Jupyter notebooks for model training and EDA
├── mlruns/                 # Local MLflow experiment tracking logs
├── src/                    # Reusable preprocessing/inference helpers (M3)
├── tests/                  # Unit tests (pytest) (M3)
├── scripts/                # Smoke test + post-deploy evaluation scripts (M4/M5)
├── app.py                  # FastAPI inference service
├── docker-compose.yml      # Local deployment using Docker Compose (M4)
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

### M3: CI Pipeline (Automated Testing + Build Verification)
- Added unit tests using pytest:
- Image preprocessing validation (shape/type)
- Inference post-processing logic validation (thresholding)
- Added GitHub Actions CI workflow to run on every PR / push:
- Installs dependencies
- Runs unit tests
- Builds Docker image to ensure container build does not break
- Files added (M3):
    - .github/workflows/ci.yml
    - src/preprocess.py, src/inference.py
    - tests/test_preprocess.py, tests/test_inference.py

### M4: Deployment (Docker Compose + Smoke Test)
- Added docker-compose.yml for repeatable local deployment
- Added a smoke test script to validate deployment:
- Calls /health
- Calls /metrics (available after M5 is enabled)
- Files added (M4):
    - docker-compose.yml
    - scripts/smoke_test.py

### M5: Monitoring (Logs + Metrics + Post-deploy Evaluation)
- Added basic monitoring capability:
- Prediction logs including label, confidence, and latency
- /metrics endpoint exposing lightweight counters (request count, failures, last latency)
- Added a post-deploy evaluation script:
- Sends a small labeled dataset to the deployed API
- Computes accuracy and exports results to CSV
- Files added (M5):
    - scripts/eval_post_deploy.py

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


## 🚀 How to Run M3, M4, and M5

### M3 — CI (Run Tests Locally)

Run unit tests using `pytest`:

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

### M3 — Verify CI on GitHub
- Push changes to GitHub
- Go to Actions tab
- Confirm workflow success (tests + docker build)

### M4 — Deployment (Docker Compose)
```bash
docker compose up --build -d
```

Verify service:
```bash
curl http://localhost:8000/health
```

### M4 — Smoke Test
```bash
python scripts/smoke_test.py http://localhost:8000
```

### M5 — Monitoring
Check metrics:
```bash
curl http://localhost:8000/metrics
```

View logs:
```bash
docker compose logs -f api
```

### M5 — Post-deploy Evaluation

Prepare dataset:
```text
eval_data/
  Cat/
  Dog/
```

Run evaluation:
```bash
python scripts/eval_post_deploy.py \
  --base-url http://localhost:8000 \
  --data eval_data
  ```
  Output:
- Prints accuracy
- Generates eval_results.csv