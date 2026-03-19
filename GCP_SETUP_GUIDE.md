# Google Cloud Run Setup Guide

This guide covers:
1. One-time Google Cloud setup
2. Manual deployment from your machine
3. Automatic deployment from GitHub Actions

## 1) One-Time Google Cloud Setup

### Prerequisites
- Google Cloud project with billing enabled
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Docker installed

### Set variables

```bash
PROJECT_ID="<your-project-id>"
REGION="asia-south1"
REPO="resnet-repo"
SERVICE="resnet-api"

gcloud config set project ${PROJECT_ID}
```

### Enable required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable iamcredentials.googleapis.com
```

### Create Artifact Registry (one-time)

```bash
gcloud artifacts repositories create ${REPO} \
  --repository-format=docker \
  --location=${REGION} \
  --description="Model deployment repo"
```

## 2) Manual Deployment to Cloud Run

### Build and push container image

```bash
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:manual-$(date +%Y%m%d%H%M%S)"

gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

### Deploy service

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

### Get service URL

```bash
gcloud run services describe ${SERVICE} \
  --region ${REGION} \
  --format='value(status.url)'
```

### Validate deployment

```bash
curl "<cloud-run-url>/"

curl -X POST "<cloud-run-url>/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/seed.jpg"
```

## 3) Automatic Deployment via GitHub Actions

Workflow file: `.github/workflows/deploy.yml`

Trigger:
- Every push to `main`

### Create deploy service account

```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name "GitHub Actions Service Account"

SA_EMAIL="github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

### Create JSON key (for current workflow)

```bash
gcloud iam service-accounts keys create key.json \
  --iam-account=${SA_EMAIL}
```

### Add GitHub repository secrets

Go to: **Repository Settings → Secrets and variables → Actions**

Add:
- `GCP_PROJECT_ID` = your GCP project ID
- `GCP_REGION` = deploy region (for example `asia-south1`)
- `GCP_SA_KEY` = full content of `key.json`

Then push to `main`:

```bash
git push origin main
```

The workflow will build an immutable image tagged with commit SHA, push it to Artifact Registry, and deploy it to Cloud Run.

## 4) Troubleshooting

- Build fails:
  - Verify `Dockerfile` and `requirements.txt`.
- Push fails:
  - Verify Artifact Registry exists in selected region.
  - Verify service account has `roles/artifactregistry.writer`.
- Deploy fails:
  - Verify `roles/run.admin` and `roles/iam.serviceAccountUser` are granted.
  - Verify Cloud Run API is enabled.
- Runtime errors:

```bash
gcloud run services logs read ${SERVICE} --region ${REGION}
```

## 5) Security Notes

- Do not commit `key.json` to Git.
- Keep service account permissions minimal.
- Use GitHub Secrets for credentials only.