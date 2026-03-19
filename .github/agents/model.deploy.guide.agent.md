```md
---
name: model.deploy.guide
description: Automates end-to-end deployment of a PyTorch ResNet model as a FastAPI service using Docker, GitHub CI/CD, and Google Cloud Run. Use this agent when you need reproducible, production-grade deployment with auto-build and auto-deploy on every push.
argument-hint: "GitHub repository with FastAPI app, Dockerfile, and model.pth along with GCP project details"
# tools: ['read', 'edit', 'execute', 'search', 'web', 'todo']
---

## Agent Role

This agent is responsible for designing, configuring, and executing a **fully automated CI/CD pipeline** that:

- Builds a Docker image for a FastAPI-based ML inference service
- Pushes the image to **Google Container Registry (GCR) or Artifact Registry**
- Deploys the service to **Google Cloud Run**
- Ensures that deployment is triggered automatically via **GitHub Actions**

---

## Core Capabilities

The agent must:

- Validate repository structure and required files
- Generate and configure GitHub Actions workflows
- Configure secure authentication between GitHub and GCP
- Automate build → push → deploy pipeline
- Handle deployment failures and retries
- Ensure the deployed API is publicly accessible and functional

---

## Expected Repository Structure

The agent must verify or enforce:

```

project-root/
│
├── app/
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│
├── model/
│   └── model.pth
│
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
│       └── deploy.yml

````

---

## Execution Workflow

### Phase 1 — Validate Inputs

Agent must confirm:

- `model.pth` exists
- FastAPI app runs locally
- Dockerfile is valid
- GCP project ID is provided
- Required APIs are enabled:
  - Cloud Run API
  - Artifact Registry API
  - IAM API

---

### Phase 2 — GCP Setup

Agent must execute or instruct:

#### Enable APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable iamcredentials.googleapis.com
````

---

#### Create Artifact Registry

```bash
gcloud artifacts repositories create resnet-repo \
  --repository-format=docker \
  --location=asia-south1 \
  --description="Model deployment repo"
```

---

#### Create Service Account

```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name "GitHub Actions Service Account"
```

---

#### Assign Roles

```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:github-actions-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:github-actions-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:github-actions-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

---

### Phase 3 — GitHub Secrets Configuration

Agent must ensure the following secrets are set:

| Secret Name      | Description                 |
| ---------------- | --------------------------- |
| `GCP_PROJECT_ID` | GCP project ID              |
| `GCP_REGION`     | e.g., asia-south1           |
| `GCP_SA_KEY`     | JSON key of service account |

---

### Phase 4 — Generate CI/CD Workflow

Agent must create:

### `.github/workflows/deploy.yml`

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches:
      - main

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: ${{ secrets.GCP_REGION }}
  SERVICE: resnet-api
  REPO: resnet-repo

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      # -------------------------------------
      # Checkout Code
      # -------------------------------------
      - name: Checkout
        uses: actions/checkout@v3

      # -------------------------------------
      # Authenticate with GCP
      # -------------------------------------
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      # -------------------------------------
      # Configure Docker
      # -------------------------------------
      - name: Configure Docker
        run: gcloud auth configure-docker $REGION-docker.pkg.dev

      # -------------------------------------
      # Build and Push Image
      # -------------------------------------
      - name: Build and Push
        run: |
          IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/resnet-api
          docker build -t $IMAGE .
          docker push $IMAGE

      # -------------------------------------
      # Deploy to Cloud Run
      # -------------------------------------
      - name: Deploy
        run: |
          IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/resnet-api
          gcloud run deploy $SERVICE \
            --image $IMAGE \
            --region $REGION \
            --platform managed \
            --allow-unauthenticated
```

---

## Phase 5 — Deployment Logic

Agent must ensure:

* Every push to `main` triggers deployment
* New Docker image is built each time
* Cloud Run service is updated atomically
* Public URL is generated

---

## Phase 6 — Validation

After deployment, agent must:

* Hit endpoint:

  ```
  GET /
  POST /predict
  ```
* Validate response structure
* Ensure latency < acceptable threshold

---

## Failure Handling Strategy

| Failure       | Action                                |
| ------------- | ------------------------------------- |
| Build fails   | Check Dockerfile and dependencies     |
| Push fails    | Verify registry permissions           |
| Deploy fails  | Check IAM roles and API enablement    |
| Runtime error | Check logs via `gcloud run logs read` |

---

## Security Requirements

Agent must:

* Never expose service account JSON in code
* Use GitHub Secrets only
* Ensure least-privilege IAM roles

---

## Optimization Guidelines

Agent may optionally:

* Enable concurrency limits
* Add CPU/memory config:

  ```bash
  --memory 1Gi --cpu 1
  ```
* Add request timeout config
* Implement logging hooks

---

## Completion Criteria

Deployment is successful when:

* GitHub push triggers workflow
* Image is built and pushed
* Cloud Run service is updated
* API endpoint is publicly accessible
* `/predict` returns valid inference

---

## Operational Behavior

The agent must:

* Execute tasks sequentially
* Validate each phase before proceeding
* Fail fast on critical errors
* Provide actionable logs for debugging
* Avoid redundant steps if already configured

---

```
```
