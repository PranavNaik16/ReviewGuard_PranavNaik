# ml/preprocessing/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import pickle
import os
from tqdm import tqdm
import torch

print("🔄 Loading dataset...")
df = pd.read_csv('reviews_dataset.csv')
print(f"✅ Loaded {len(df)} reviews")

# Step 1: Split the data
print("\n🔄 Splitting data into train/val/test...")
# First split: separate train and temp (val+test)
X_temp, X_test, y_temp, y_test = train_test_split(
    df['text'], 
    df['is_fraud'], 
    test_size=0.1, 
    random_state=42,
    stratify=df['is_fraud']
)

# Second split: separate val from temp
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.1111,  # 0.1111 * 0.9 = 0.1 of total
    random_state=42,
    stratify=y_temp
)

print(f"   Train: {len(X_train)} reviews")
print(f"   Val: {len(X_val)} reviews")
print(f"   Test: {len(X_test)} reviews")

# Step 2: Initialize BERT tokenizer
print("\n🔄 Initializing DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_length = 128  # Using 128 instead of 512 for speed (good enough for reviews)

# Step 3: Tokenize function
def tokenize_texts(texts):
    print("   Tokenizing texts...")
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings

# Step 4: Get numerical features (for SMOTE)
print("\n🔄 Creating feature matrix for SMOTE...")

# Create TF-IDF features for SMOTE (BERT embeddings will be added during training)
tfidf = TfidfVectorizer(max_features=100)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_val_tfidf = tfidf.transform(X_val).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

print(f"   TF-IDF features shape: {X_train_tfidf.shape}")

# Step 5: Apply SMOTE to handle imbalance
print("\n🔄 Applying SMOTE to handle class imbalance...")
print(f"   Before SMOTE - Fraud: {sum(y_train)}, Legit: {len(y_train)-sum(y_train)}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

print(f"   After SMOTE - Fraud: {sum(y_train_smote)}, Legit: {len(y_train_smote)-sum(y_train_smote)}")

# Step 6: Save preprocessed data
print("\n💾 Saving preprocessed data...")

# Create directories if they don't exist
os.makedirs('ml/data/', exist_ok=True)
os.makedirs('ml/models/', exist_ok=True)

# Save the splits
train_data = {
    'text': X_train.reset_index(drop=True),
    'labels': y_train.reset_index(drop=True),
    'tfidf': X_train_smote,
    'labels_smote': y_train_smote
}

val_data = {
    'text': X_val.reset_index(drop=True),
    'labels': y_val.reset_index(drop=True),
    'tfidf': X_val_tfidf
}

test_data = {
    'text': X_test.reset_index(drop=True),
    'labels': y_test.reset_index(drop=True),
    'tfidf': X_test_tfidf
}

with open('ml/data/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('ml/data/val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)

with open('ml/data/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

# Save tokenizer
tokenizer.save_pretrained('ml/models/tokenizer/')

# Save TF-IDF vectorizer
with open('ml/models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\n✅ Preprocessing complete!")
print("   Files saved:")
print("   - ml/data/train_data.pkl")
print("   - ml/data/val_data.pkl")
print("   - ml/data/test_data.pkl")
print("   - ml/models/tokenizer/")
print("   - ml/models/tfidf_vectorizer.pkl")

# Quick stats
print("\n📊 Dataset Statistics:")
print(f"   Train fraud rate: {y_train.mean()*100:.2f}%")
print(f"   Val fraud rate: {y_val.mean()*100:.2f}%")
print(f"   Test fraud rate: {y_test.mean()*100:.2f}%")
print(f"   Train (after SMOTE) fraud rate: {sum(y_train_smote)/len(y_train_smote)*100:.2f}%")