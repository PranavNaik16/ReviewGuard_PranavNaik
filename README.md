# 🛡️ Review Guard - Fraudulent Review Detection System

An AI-powered backend service that detects fraudulent reviews using NLP and behavioral heuristics. Built for high-scale e-commerce platforms (inspired by Amazon/Flipkart's spam detection systems).

## 📋 Table of Contents
- [Features](#-features)
- [Performance Benchmarks](#-performance-benchmarks)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Model Files](#-model-files)
- [Training Instructions](#-training-instructions)
- [API Documentation](#-api-documentation)
- [Sample API Responses](#-sample-api-responses)
- [Drift Monitoring](#-drift-monitoring)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Assumptions](#-assumptions)
- [Evaluation Criteria](#-evaluation-criteria)
- [Author](#-author)

## 🎯 Features

- **NLP-Powered Detection**: Fine-tuned DistilBERT model for text analysis
- **Behavioral Heuristics**: Velocity-based anomaly detection (posts per day)
- **Imbalance Handling**: SMOTE oversampling for 95/5 class distribution
- **High-Speed Inference**: ONNX + INT8 quantization for <200ms latency
- **Batch Processing**: Handle 100+ reviews per request
- **Real-time Monitoring**: KS-test drift detection with alerts
- **Explainable AI**: Feature importance with explanations
- **Production Ready**: MongoDB persistence, Redis caching, Docker support
- **Interactive Dashboard**: Simple HTML/CSS/JS frontend for testing

## 📊 Performance Benchmarks

### Inference Speed (Quantized ONNX)
Tested on CPU with batch processing:

| Batch Size | Total Time (ms) | Per Review (ms) | Requirement |
|------------|-----------------|-----------------|-------------|
| 1 | 54.76 | **54.76** | ✅ <200ms |
| 8 | 363.53 | **45.44** | ✅ <200ms |
| 16 | 755.91 | **47.24** | ✅ <200ms |
| 32 | 1621.65 | **50.68** | ✅ <200ms |
| 64 | 3220.85 | **50.33** | ✅ <200ms |

### Model Performance

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| AUC-ROC | **1.000** | >0.92 | ✅ EXCEEDED |
| F1 Score (Fraud) | **1.000** | >0.85 | ✅ EXCEEDED |
| Training Data | 50,000 reviews | - | ✅ Generated |
| Fraud Rate | 5.15% | ~5% | ✅ Imbalanced |

## 🏗️ Architecture


                     ┌──────────────────────────┐
                     │        Client Layer       │
                     │   Web App / Mobile App    │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │       API Layer           │
                     │        FastAPI            │
                     │   Backend Orchestration   │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      Inference Engine     │
                     │     ONNX Runtime (INT8)   │
                     │   DistilBERT Fraud Model  │
                     └─────────────┬─────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
     ┌────────────────────┐                 ┌────────────────────┐
     │      MongoDB        │                 │        Redis        │
     │   Review Storage    │                 │    Velocity Cache   │
     │   Audit Logs        │                 │   User Rate Cache   │
     └──────────┬──────────┘                 └──────────┬──────────┘
                │                                       │
                └──────────────┬────────────────────────┘
                               ▼
                     ┌──────────────────────────┐
                     │   Monitoring & Analytics │
                     │  Drift Detection (KS)    │
                     │  Velocity Tracking       │
                     │  Fraud Score Monitoring  │
                     └──────────────────────────┘


## 💻 Tech Stack

### Backend
- **FastAPI** - High-performance async API framework
- **Uvicorn** - ASGI server
- **MongoDB** - Review storage with indexes
- **Redis** - Velocity caching (24h TTL)
- **Pytest** - Unit testing with 88% coverage

### Machine Learning
- **Hugging Face Transformers** - DistilBERT model
- **PyTorch** - Deep learning framework
- **ONNX Runtime** - Optimized inference
- **Scikit-learn** - SMOTE, preprocessing
- **Imbalanced-learn** - SMOTE for handling class imbalance

### Frontend
- **HTML/CSS/JavaScript** - Simple dashboard
- **Fetch API** - REST API integration

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (for MongoDB and Redis)
- 8GB RAM minimum

### Installation

```bash
# Clone repository
git clone https://github.com/PranavNaik16/ReviewGuard_PranavNaik.git
cd ReviewGuard_PranavNaik

# Install dependencies
pip install -r backend/requirements.txt

# Start MongoDB & Redis with Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
docker run -d -p 6379:6379 --name redis redis:latest

# Run the API
cd backend/api
uvicorn main:app --reload

Run the Frontend Dashboard
Simply open frontend/index.html in your browser or use Live Server in VS Code.

The dashboard will connect to your API at http://localhost:8000
```

## 📦 Model Files

The trained model files are **not included in the GitHub repository** due to file size limitations (>50 MB). 

### To get the model files:

**Option 1: Generate them yourself**
```bash
# Run these commands to create all model files
python ml/generate_data.py
python ml/preprocessing/preprocess.py
python ml/training/train.py
python ml/export/convert_to_onnx.py
```
**Option 2: Download model files from Google Drive**
```bash
Click here to download all model files : (https://drive.google.com/drive/folders/1MwDD241H39wSH48wmBAZJjdSl05gyNzN?usp=sharing)
```

### 🧠 Model Performance Summary

| Model | Size | Speed | Use Case |
|------|------|------|------|
| `best_model.pt` | 253 MB | - | Training / Retraining |
| `model.onnx` | 253 MB | ~100ms | Standard inference |
| `model_quantized.onnx` | 63 MB | ~50ms | Production API |

- `tokenizer/` - BERT tokenizer files

> **Note:** The quantized model (`model_quantized.onnx`) is used by the API for **fast CPU inference**.

## 🏋️ Training Instructions

### Step 1: Generate Dataset

Run the synthetic data generator to create the review dataset.

```bash
python ml/generate_data.py
```

**Output**

```
reviews_dataset.csv
```

- Contains **50,000 reviews**
- **5% fraudulent reviews**
- **95% legitimate reviews**

---

### Step 2: Preprocess Data with SMOTE

This step cleans the text, tokenizes it for BERT, and handles class imbalance using **SMOTE oversampling**.

```bash
python ml/preprocessing/preprocess.py
```

**Outputs**

```
train_data.pkl
val_data.pkl
test_data.pkl
```

Dataset split:

- **80% Training**
- **10% Validation**
- **10% Test**

---

### Step 3: Train Model

Train the fraud detection model using **DistilBERT**.

#### Local Training (CPU)

```bash
python ml/training/train.py
```

Estimated time:

```
~1.5 hours
```

#### Google Colab Training (GPU)

1. Upload the following files to Colab:

```
train_data.pkl
val_data.pkl
```

2. Run the training notebook.

Estimated time:

```
~20 minutes
```

The training process outputs the best performing model:

```
best_model.pt
```

---

### Step 4: Export Model to ONNX

Convert the trained PyTorch model into **ONNX format** and apply **INT8 quantization** for faster CPU inference.

```bash
python ml/export/convert_to_onnx.py
```

**Outputs**

```
model.onnx
model_quantized.onnx
```

Performance improvement:

```
74.9% model size reduction
```

The **quantized ONNX model** is used by the **FastAPI inference service** for production.

## 📡 API Documentation

Once the server is running, open the interactive Swagger UI:

```
http://localhost:8000/docs
```

This provides a full interface to test all API endpoints.

---

### Available Endpoints

| Method | Endpoint | Description |
|------|------|------|
| **POST** | `/api/reviews/detect/batch` | Batch fraud detection (supports up to **100 reviews per request**) |
| **POST** | `/api/reviews/submit` | Submit a single review for fraud analysis |
| **GET** | `/api/reviews/{id}` | Retrieve review details and fraud score by ID |
| **GET** | `/api/drift/status` | View model drift monitoring status and statistics |
| **GET** | `/api/health` | API health check (model, database, and cache status) |

---

### Swagger Testing Interface

The Swagger UI allows you to:

- Test endpoints directly from the browser
- View request/response schemas
- Execute batch fraud detection requests
- Monitor API responses in real time

Open:

```
http://localhost:8000/docs
```

to explore the full API.

## 📈 Drift Monitoring

The system continuously monitors model performance using the **Kolmogorov-Smirnov (KS) statistical test** to detect **data distribution drift** between training data and live predictions.

---

### ⚙️ How It Works

- **Frequency:** Runs every **1 hour** using a background cron job
- **Method:** Compares **live prediction score distribution** with the **training score distribution**
- **Statistical Test:** Kolmogorov-Smirnov (KS Test)
- **Drift Threshold:**  

```
p-value < 0.05
```

If the threshold is crossed, the system triggers a **drift alert**.

- **Storage:** Drift events are logged in **MongoDB**

---

### 📡 Drift Status Endpoint

```
GET /api/drift/status
```

Example Response:

```json
{
  "drift_detected": false,
  "current_p_value": 0.234,
  "recent_drifts": 0,
  "samples_analyzed": 1250,
  "drift_events": []
}
```

---

### ⚠️ Drift Alert Example (p < 0.05)

When the system detects drift:

```json
{
  "drift_detected": true,
  "current_p_value": 0.023,
  "recent_drifts": 1,
  "drift_events": [
    {
      "timestamp": "2024-03-12T10:30:00Z",
      "ks_statistic": 0.156,
      "p_value": 0.023,
      "sample_size": 500
    }
  ]
}
```

---

### 🧠 Why Drift Monitoring Matters

Over time, fraud patterns may evolve (e.g., **AI-generated spam reviews**).  
Drift monitoring helps detect when the model's training data **no longer represents real-world data**, signaling the need for **model retraining**.
