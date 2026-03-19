# Google Cloud Setup Guide

To get the deployment working cleanly with the GitHub Actions pipeline, we need to create the Google Cloud project, enable the required APIs, set up an Artifact Registry (to store your Docker Image), and create a Service Account for GitHub so it could deploy the image.

Here is the step-by-step guide to run either through your Google Cloud Console or using the `gcloud` CLI (which is usually much faster).

### Step 1: Initialize the Project and Enable APIs

Open the Google Cloud Shell terminal in your console (the `>_` icon at the top right) or run this on your local machine if you have the `gcloud` CLI installed.

```bash
# 1. Set your Project ID (Replace with your actual project ID)
PROJECT_ID="synergy-490715"
gcloud config set project $PROJECT_ID

# 2. Enable Required APIs (Cloud Run, Artifact Registry, IAM)
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable iamcredentials.googleapis.com
```

### Step 2: Create an Artifact Registry Repository

This acts as the Docker container storage inside Google Cloud where GitHub Actions will push the built images.

```bash
# Choose a region, e.g., asia-south1, us-central1, etc.
REGION="asia-south1"

gcloud artifacts repositories create resnet-repo \
  --repository-format=docker \
  --location=$REGION \
  --description="Model deployment repo"
```

### Step 3: Create a Service Account for GitHub Actions

We need to create a specific identity (service account) that gives GitHub the precise permissions needed to push the image and start Cloud Run.

```bash
# 1. Create the Service Account
gcloud iam service-accounts create github-actions-sa \
  --display-name "GitHub Actions Service Account"

# Get the full email of the new service account
SA_EMAIL="github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 2. Grant Cloud Run Admin Role (Allows deploying the application)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.admin"

# 3. Grant Artifact Registry Writer Role (Allows pushing Docker images)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer"

# 4. Grant IAM Service Account User Role (Run needs this to execute the container)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

### Step 4: Generate the JSON Key

We need to generate a key file for this Service Account, which we will provide to GitHub safely as a secret.

```bash
# Generate the key and save it to a file named 'key.json'
gcloud iam service-accounts keys create key.json \
  --iam-account=${SA_EMAIL}

# Print the key contents to the console so you can copy it
cat key.json
```

### Step 5: Configure GitHub Secrets

Take the contents from the `key.json` file and the other variables to your GitHub Repository Settings.

1. Go to your repository on GitHub.
2. Click **Settings** > **Secrets and variables** > **Actions** > **New repository secret**.
3. Add the following secrets carefully:

| Secret Name | What to set it to |
| :--- | :--- |
| `GCP_PROJECT_ID` | Your actual Project ID (e.g. `my-awesome-gcp-project-123`) |
| `GCP_REGION` | The region you used for the Artifact Registry (e.g. `asia-south1` or `us-central1`) |
| `GCP_SA_KEY` | Paste the **entire content** of the `key.json` file you generated above. |

Once this is configured, pushing the code to the `main` branch will automatically launch the deployed FastAPI backend!
