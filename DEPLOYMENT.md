# Deployment Guide

This guide explains how to deploy the Churn Prediction System to the cloud.

## Architecture
- **Backend (API)**: FastAPI app hosted on **Render** (or Railway/Heroku).
- **Frontend (Dashboard)**: Streamlit app hosted on **Streamlit Cloud**.
- **Database**: MongoDB hosted on **MongoDB Atlas**.

---

## Prerequisites
1. **GitHub Account**: Push this entire repository to a new public or private GitHub repository.
2. **MongoDB Atlas**: You already have this set up. Keep your connection string handy.

---

## Step 1: Deploy the Backend (API) to Render

1. Create an account on [Render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Configure the service:
   - **Name**: `churn-api` (or similar)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.
6. Wait for the deployment to finish. Copy the **URL** (e.g., `https://churn-api.onrender.com`).

---

## Step 2: Configure Automated Scheduling (GitHub Actions)

We use GitHub Actions to run the training, monitoring, and inference pipelines automatically every 15 minutes for free.

1. Go to your GitHub repository settings.
2. Navigate to **Secrets and variables** > **Actions**.
3. Click **New repository secret**.
4. Add the following secret:
   - **Name**: `MONGO_URI`
   - **Value**: Your MongoDB connection string (same as in Step 3).

The workflow file is located at `.github/workflows/mlops-pipeline.yml`. It will:
- Retrain the model.
- Check for data drift.
- Update customer risk scores in the database.
- Commit any new models or reports back to the repository.

---

## Step 3: Deploy the Frontend (Dashboard) to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **New app**.
3. Select your GitHub repository.
4. Configure the settings:
   - **Main file path**: `dashboard/crm.py`
5. Click **Advanced settings** (or "Secrets" in the dashboard settings after creation) and add the following secrets:

```toml
# .streamlit/secrets.toml

API_URL = "https://your-render-app-name.onrender.com"
MONGO_URI = "mongodb+srv://kerim-mlops:kerim-mlops@cluster0.vm3zsx5.mongodb.net/"
```

6. Click **Deploy**.

---

## Step 3: Verify

1. Open your Streamlit App URL.
2. The app should connect to your MongoDB Atlas database (data will persist).
3. When you run predictions, it will send requests to your Render API.

## Troubleshooting

- **API Sleeping**: On the free tier, Render spins down the API after inactivity. The first request might take 50+ seconds.
- **Missing Files**: Ensure `models/churn_model.pkl` is in your GitHub repo.
