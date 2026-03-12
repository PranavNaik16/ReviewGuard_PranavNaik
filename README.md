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
[**Click here to download all model files**]([https://drive.google.com/your-link-here](https://drive.google.com/drive/folders/1MwDD241H39wSH48wmBAZJjdSl05gyNzN?usp=sharing))
```

### 🧠 Model Performance Summary

| Model | Size | Speed | Use Case |
|------|------|------|------|
| `best_model.pt` | 253 MB | - | Training / Retraining |
| `model.onnx` | 253 MB | ~100ms | Standard inference |
| `model_quantized.onnx` | 63 MB | ~50ms | Production API |

- `tokenizer/` - BERT tokenizer files

> **Note:** The quantized model (`model_quantized.onnx`) is used by the API for **fast CPU inference**.

