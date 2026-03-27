# How This Is Implemented

This document explains how the Paddy Seed classifier API is implemented, how to run it locally, how Docker is prepared, and how deployment to Google Cloud Run works.

## 1) Implementation Overview

### FastAPI service
- Entrypoint: `app/app.py`
- Framework: FastAPI
- Endpoints:
  - `GET /` returns service health/basic info.
  - `POST /predict` accepts an image (`multipart/form-data`) and returns prediction + confidence.

### Model loading and inference
- Model code: `app/model.py`
- The model is loaded lazily on first `POST /predict` call.
- A thread lock prevents duplicate model loads under concurrent requests.
- Model path resolution order:
  1. `MODEL_PATH` environment variable (if set)
  2. `model/paddy_seed_model_final.pth` in this repo
  3. `/workspace/model/paddy_seed_model_final.pth` in container
- Checkpoint compatibility:
  - Supports checkpoints containing a nested `state_dict`.
  - Supports direct raw `state_dict` files.
  - Uses checkpoint metadata (`architecture`, `class_names`, `num_classes`) when available.

### Image preprocessing
Input images are converted to RGB, resized to `224x224`, converted to tensor, then normalized with ImageNet stats:
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

## 2) Run Locally

From project root:

```bash
pip install -r requirements.txt
python run.py
```

Open:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/`

Optional env variables:

```bash
# Windows PowerShell example
$env:HOST="127.0.0.1"
$env:PORT="8000"
$env:RELOAD="true"
python run.py
```

## 3) Docker Preparation

### Build image

```bash
docker build -t paddy-seed-api:local .
```

### Run container locally

```bash
docker run --rm -p 8000:8080 paddy-seed-api:local
```

Then test at:
- `http://127.0.0.1:8000/docs`

Note:
- The container listens on port `8080` internally (Cloud Run-compatible).
- The model file is included from the repo `model/` directory.

## 4) Manual Deployment to Cloud Run

### Prerequisites
- `gcloud` CLI installed and authenticated
- Docker installed
- GCP project with billing enabled

### Step A: Set variables

```bash
PROJECT_ID="<your-project-id>"
REGION="asia-south1"
SERVICE="resnet-api"
REPO="resnet-repo"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:manual-$(date +%Y%m%d%H%M%S)"

gcloud config set project ${PROJECT_ID}
```

### Step B: Enable required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable iamcredentials.googleapis.com
```

### Step C: Create Artifact Registry (one-time)

```bash
gcloud artifacts repositories create ${REPO} \
  --repository-format=docker \
  --location=${REGION} \
  --description="Model deployment repo"
```

### Step D: Build and push image

```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

### Step E: Deploy to Cloud Run

```bash
gcloud run deploy ${SERVICE} \
  --image ${IMAGE_URI} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1
```

Get service URL:

```bash
gcloud run services describe ${SERVICE} \
  --region ${REGION} \
  --format='value(status.url)'
```

## 5) Auto-Deploy with GitHub Actions

Workflow file: `.github/workflows/deploy.yml`

Trigger:
- Push to `main`

Required GitHub repository secrets:
- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_SA_KEY` (JSON key for deploy service account)

Pipeline behavior:
1. Checkout code
2. Authenticate to GCP
3. Configure Docker auth for Artifact Registry
4. Build and push Docker image
5. Deploy new revision to Cloud Run

## 6) Post-Deployment Validation

Check service root:

```bash
curl "<cloud-run-url>/"
```

Run prediction test:

```bash
curl -X POST "<cloud-run-url>/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/seed.jpg"
```

If a deployment fails, inspect logs:

```bash
gcloud run services logs read ${SERVICE} --region ${REGION}
```