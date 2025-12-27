# 🛡️ ChurnGuard CRM - End-to-End MLOps Pipeline

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.0-0194E2.svg?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Prefect](https://img.shields.io/badge/Prefect-2.19.0-070E2B.svg?style=flat&logo=prefect&logoColor=white)](https://www.prefect.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-24.0.5-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**ChurnGuard CRM** is a production-grade MLOps project designed to predict customer churn in real-time. It moves beyond experimental notebooks to a fully automated, modular engineering system featuring continuous training, data drift monitoring, and a user-friendly CRM dashboard.

---

## 🏗️ Architecture

The system is built on a modular architecture separating data ingestion, feature engineering, model training, and inference.

```mermaid
graph TD
    subgraph "Data Pipeline"
        Raw[Raw Data] --> Ingest[Ingestion Task]
        Ingest --> Process[Preprocessing & Validation]
        Process --> Feature[Feature Engineering]
    end

    subgraph "ML Pipeline (Prefect)"
        Feature --> Train[Model Training (XGBoost)]
        Train --> Eval[Evaluation]
        Eval --> Registry[MLflow Model Registry]
    end

    subgraph "Monitoring"
        Registry --> Drift[Evidently AI Drift Detection]
        Drift --> Alert[Alerts/Reports]
    end

    subgraph "Serving & UI"
        Registry --> API[FastAPI Inference Service]
        DB[(MongoDB)] --> Dashboard[Streamlit CRM]
        API --> Dashboard
    end
```

## ✨ Key Features

- **🤖 Automated Workflows:** Orchestrated via **Prefect** and **GitHub Actions** for scheduled retraining and batch inference.
- **🛡️ Robust Engineering:** Strict type checking (Pydantic), modular code structure, and comprehensive unit/integration tests.
- **🧪 Experiment Tracking:** Full integration with **MLflow** to track parameters, metrics, and artifacts.
- **📉 Observability:** Automated data drift detection using **Evidently AI** to monitor model performance over time.
- **⚡ Real-time Inference:** High-performance REST API built with **FastAPI**.
- **📊 Interactive Dashboard:** A **Streamlit**-based CRM interface for visualizing customer risk profiles.

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.9 | Core programming language |
| **Modeling** | Scikit-learn, XGBoost | Machine learning algorithms |
| **API** | FastAPI | High-performance web framework for building APIs |
| **Orchestration** | Prefect | Workflow orchestration and scheduling |
| **Tracking** | MLflow | Experiment tracking and model registry |
| **Monitoring** | Evidently AI | Data drift and model performance monitoring |
| **Database** | MongoDB | NoSQL database for storing customer profiles |
| **Dashboard** | Streamlit | Interactive web app for the CRM interface |
| **CI/CD** | GitHub Actions | Automated testing and deployment pipelines |
| **Containerization** | Docker | Application containerization |

## 📂 Project Structure

```
churn-mlops/
├── .github/workflows/    # CI/CD pipelines (GitHub Actions)
├── configs/              # Configuration files (YAML)
├── dashboard/            # Streamlit CRM application
├── data/                 # Data directory (raw, processed)
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── api/              # FastAPI application
│   ├── data/             # Data loaders and validators
│   ├── features/         # Feature engineering logic
│   ├── models/           # Training and prediction logic
│   ├── monitoring/       # Drift detection (Evidently)
│   └── utils/            # Utilities (logging, config)
├── tests/                # Unit and integration tests
├── Dockerfile            # Docker configuration
├── pyproject.toml        # Project dependencies
├── requirements.txt      # Python dependencies
└── run_*.py              # Entry points for flows
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- MongoDB (local or Atlas)
- Docker (optional)

### 1. Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abdulkerimosman/churn-mlops.git
    cd churn-mlops
    ```

2.  **Create and activate virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\Activate.ps1
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    MONGO_URI=mongodb://localhost:27017/
    API_URL=http://localhost:8000
    ```

### 2. Running the Pipeline

**Start MLflow Server:**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 8080
```

**Run Training Flow:**
```bash
python run_training_flow.py
```

**Run API:**
```bash
uvicorn src.api.app:app --reload
```

**Run Dashboard:**
```bash
streamlit run dashboard/crm.py
```

### 3. Docker Setup

Build and run the entire stack using Docker:

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## 🧪 Testing

Run the test suite to ensure system stability:

```bash
pytest tests/unit/ -v --cov=src --cov-report=html
```

## 🔄 MLOps Automation

The project uses **GitHub Actions** to automate the lifecycle:

- **Schedule:** Runs every 15 minutes.
- **Steps:**
    1.  **Training:** Retrains the model on the latest data.
    2.  **Monitoring:** Checks for data drift using Evidently.
    3.  **Inference:** Runs batch predictions and updates the MongoDB database.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

