# backend/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import onnxruntime as ort
import numpy as np
from transformers import DistilBertTokenizer
import redis
import pymongo
from datetime import datetime, timedelta
import os
import uuid
import json
import asyncio
from scipy import stats

# Add after imports, before app initialization
import sys
from pathlib import Path

def get_project_root():
    """Get absolute path to project root regardless of how script is run"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys.executable).parent.parent
    else:
        # Running as script
        return Path(__file__).parent.parent.parent

PROJECT_ROOT = get_project_root()

# Initialize FastAPI
app = FastAPI(title="Review Guard API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODELS ==========
class ReviewSubmit(BaseModel):
    text: str
    user_id: int
    rating: int

class ReviewBatch(BaseModel):
    reviews: List[ReviewSubmit]

class DetectionResult(BaseModel):
    review_id: str
    score: float
    explanation: List[str]
    status: str

# ========== INITIALIZATION ==========
print("🚀 Starting Review Guard API...")

# Set correct paths for models
tokenizer_path = PROJECT_ROOT / 'ml' / 'models' / 'tokenizer'
model_path = PROJECT_ROOT / 'ml' / 'models' / 'model_quantized.onnx'

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Tokenizer path: {tokenizer_path}")
print(f"📁 Model path: {model_path}")

# Load tokenizer
if tokenizer_path.exists():
    tokenizer = DistilBertTokenizer.from_pretrained(str(tokenizer_path))
    print("✅ Tokenizer loaded from local path")
else:
    print(f"❌ Tokenizer not found at {tokenizer_path}")
    print("⚠️ Falling back to downloading from HuggingFace...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load ONNX model
if model_path.exists():
    ort_session = ort.InferenceSession(str(model_path))
    print("✅ ONNX model loaded from local path")
else:
    print(f"❌ Model not found at {model_path}")
    print("⚠️ Please ensure model file exists")
    raise FileNotFoundError(f"Model not found at {model_path}")

# Connect to MongoDB
try:
    mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = mongo_client['review_guard']
    reviews_collection = db['reviews']
    drift_collection = db['drift_logs']
    # Create indexes
    reviews_collection.create_index([('user_id', pymongo.ASCENDING)])
    reviews_collection.create_index([('timestamp', pymongo.DESCENDING)])
    reviews_collection.create_index([('score', pymongo.DESCENDING)])
    print("✅ MongoDB connected")
except:
    print("⚠️ MongoDB not available - running without database")
    db = None

# Connect to Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected")
except:
    print("⚠️ Redis not available - running without cache")
    redis_client = None

# Store training scores for drift detection
training_scores = np.random.beta(2, 50, 10000)  # Mock training distribution
drift_threshold = 0.05

# ========== HELPER FUNCTIONS ==========
def get_user_velocity(user_id: int) -> int:
    """Get user's posting velocity from Redis cache"""
    if redis_client:
        key = f"user_velocity:{user_id}"
        velocity = redis_client.get(key)
        if velocity:
            return int(velocity)
    return np.random.poisson(2)  # Default

def update_user_velocity(user_id: int):
    """Update user's velocity in Redis"""
    if redis_client:
        key = f"user_velocity:{user_id}"
        redis_client.incr(key)
        redis_client.expire(key, 86400)  # 24 hours

def predict_review(text: str, user_id: int) -> tuple:
    """Run inference on a single review"""
    # Get velocity
    velocity = get_user_velocity(user_id)
    
    # Tokenize
    encodings = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Run ONNX inference
    ort_inputs = {
        'input_ids': encodings['input_ids'].numpy().astype(np.int64),
        'attention_mask': encodings['attention_mask'].numpy().astype(np.int64),
        'velocity': np.array([velocity], dtype=np.float32)
    }
    
    outputs = ort_session.run(None, ort_inputs)
    probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
    fraud_prob = float(probs[0][1])
    
    # Generate explanation
    explanation = []
    if fraud_prob > 0.7:
        explanation.append("High fraud probability")
    if velocity > 5:
        explanation.append(f"High posting velocity ({velocity} posts/day)")
    
    return fraud_prob, explanation, velocity

def check_drift(live_scores: List[float]):
    """Check for data drift using KS-test"""
    if len(live_scores) < 100:
        return False, 1.0
    
    ks_statistic, p_value = stats.ks_2samp(training_scores, live_scores)
    
    if p_value < drift_threshold:
        # Log drift event
        if db:
            drift_collection.insert_one({
                'timestamp': datetime.utcnow(),
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'sample_size': len(live_scores)
            })
        print(f"⚠️ DRIFT DETECTED! p-value: {p_value:.4f}")
        return True, p_value
    
    return False, p_value

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    return {"message": "Review Guard API", "status": "running"}

@app.post("/api/reviews/submit", response_model=DetectionResult)
async def submit_review(review: ReviewSubmit, background_tasks: BackgroundTasks):
    """Submit a single review for detection"""
    # Generate review ID
    review_id = str(uuid.uuid4())
    
    # Get prediction
    score, explanation, velocity = predict_review(review.text, review.user_id)
    
    # Determine status
    status = "quarantined" if score > 0.7 else "approved"
    
    # Update velocity in background
    background_tasks.add_task(update_user_velocity, review.user_id)
    
    # Store in MongoDB
    if db is not None:
        review_doc = {
            'review_id': review_id,
            'user_id': review.user_id,
            'text': review.text,
            'rating': review.rating,
            'score': score,
            'velocity': velocity,
            'explanation': explanation,
            'status': status,
            'timestamp': datetime.utcnow()
        }
        reviews_collection.insert_one(review_doc)
    
    return DetectionResult(
        review_id=review_id,
        score=score,
        explanation=explanation,
        status=status
    )

@app.post("/api/reviews/detect/batch")
async def detect_batch(batch: ReviewBatch, background_tasks: BackgroundTasks):
    """Process a batch of reviews"""
    results = []
    live_scores = []
    
    for review in batch.reviews:
        review_id = str(uuid.uuid4())
        score, explanation, velocity = predict_review(review.text, review.user_id)
        
        live_scores.append(score)
        
        status = "quarantined" if score > 0.7 else "approved"
        
        results.append({
            'review_id': review_id,
            'score': score,
            'explanation': explanation,
            'status': status
        })
        
        # Update velocity in background
        background_tasks.add_task(update_user_velocity, review.user_id)
    
    # Check for drift periodically
    if len(live_scores) >= 100:
        drift_detected, p_value = check_drift(live_scores)
        if drift_detected:
            print(f"⚠️ Batch drift detected! p-value: {p_value:.4f}")
    
    return {"results": results}

@app.get("/api/reviews/{review_id}")
async def get_review(review_id: str):
    """Get review details by ID"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    review = reviews_collection.find_one({'review_id': review_id}, {'_id': 0})
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return review

@app.get("/api/drift/status")
async def drift_status():
    """Get current drift monitoring status"""
    if db is None:
        return {"status": "Database not available", "drift_detected": False}
    
    # Get recent drift events
    recent_drifts = list(drift_collection.find(
        {'timestamp': {'$gte': datetime.utcnow() - timedelta(hours=24)}}
    ).sort('timestamp', -1).limit(10))
    
    # Get recent scores
    recent_reviews = list(reviews_collection.find(
        {}, {'score': 1, '_id': 0}
    ).sort('timestamp', -1).limit(1000))
    
    recent_scores = [r['score'] for r in recent_reviews if 'score' in r]
    
    if len(recent_scores) >= 100:
        drift_detected, p_value = check_drift(recent_scores)
    else:
        drift_detected = False
        p_value = 1.0
    
    return {
        'drift_detected': drift_detected,
        'current_p_value': p_value,
        'recent_drifts': len(recent_drifts),
        'drift_events': recent_drifts,
        'samples_analyzed': len(recent_scores)
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    status = {
        'api': 'healthy',
        'model': 'loaded',
        'database': False,
        'redis': False
    }
    
    if db is not None:
        try:
            db.command('ping')
            status['database'] = True
        except:
            pass
    
    if redis_client:
        try:
            redis_client.ping()
            status['redis'] = True
        except:
            pass
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)