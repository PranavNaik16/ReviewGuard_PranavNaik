# ml/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertModel, DistilBertTokenizer
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from tqdm import tqdm
import os
import json

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Custom Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, velocities=None):
        self.texts = texts
        # Convert boolean labels to integers (0 or 1)
        self.labels = labels.astype(int) if hasattr(labels, 'astype') else np.array(labels, dtype=int)
        self.velocities = velocities if velocities is not None else np.zeros(len(texts))
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx],  # Now this will be int 0 or 1
            'velocity': self.velocities[idx]
        }

# Custom Model with BERT + Velocity
class FraudDetectionModel(nn.Module):
    def __init__(self, bert_model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        # BERT hidden size is 768, plus 1 velocity feature
        self.classifier = nn.Linear(768 + 1, 2)  # 2 classes: fraud/legit
        
    def forward(self, input_ids, attention_mask, velocity):
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch, 768)
        
        # Concatenate velocity feature
        velocity = velocity.unsqueeze(1)  # Shape: (batch, 1)
        combined = torch.cat([cls_embedding, velocity], dim=1)  # Shape: (batch, 769)
        
        # Apply dropout and classification
        combined = self.dropout(combined)
        output = self.classifier(combined)  # Shape: (batch, 2)
        
        return output

# Collate function for DataLoader
def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)  # Use long for classification
    velocities = torch.tensor([item['velocity'] for item in batch], dtype=torch.float32)
    
    return {
        'texts': texts,
        'labels': labels,
        'velocities': velocities
    }

print("📂 Loading preprocessed data...")
with open('ml/data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('ml/data/val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('ml/models/tokenizer/')

# Get velocities from original dataframe
df = pd.read_csv('reviews_dataset.csv')

# Fix indices - ensure we're using the right indices
train_indices = train_data['text'].index if hasattr(train_data['text'], 'index') else range(len(train_data['text']))
val_indices = val_data['text'].index if hasattr(val_data['text'], 'index') else range(len(val_data['text']))

train_velocities = df.iloc[train_indices]['velocity'].values if len(train_indices) > 0 else np.zeros(len(train_data['text']))
val_velocities = df.iloc[val_indices]['velocity'].values if len(val_indices) > 0 else np.zeros(len(val_data['text']))

print("🔄 Creating datasets...")
train_dataset = ReviewDataset(
    texts=train_data['text'].values,
    labels=train_data['labels_smote'].values,  # Using SMOTE-balanced labels
    velocities=train_velocities
)

val_dataset = ReviewDataset(
    texts=val_data['text'].values,
    labels=val_data['labels'].values,  # Original validation labels
    velocities=val_velocities
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"   Training fraud samples: {sum(train_dataset.labels)}")
print(f"   Validation fraud samples: {sum(val_dataset.labels)}")

# Initialize model
print("\n🤖 Initializing model...")
model = FraudDetectionModel().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3
best_val_f1 = 0

print("\n🎯 Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for batch in progress_bar:
        # Tokenize texts
        encodings = tokenizer(
            batch['texts'],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        # Move data to device
        labels = batch['labels'].to(device)
        velocities = batch['velocities'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask'],
            velocity=velocities
        )
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate training metrics
    train_f1 = f1_score(train_labels, train_preds, average='binary')
    train_auc = roc_auc_score(train_labels, train_preds)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
            encodings = tokenizer(
                batch['texts'],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            labels = batch['labels'].to(device)
            velocities = batch['velocities'].to(device)
            
            outputs = model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                velocity=velocities
            )
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # Calculate validation metrics
    val_f1 = f1_score(val_labels, val_preds, average='binary')
    val_auc = roc_auc_score(val_labels, val_preds)
    
    print(f"\n📊 Epoch {epoch+1} Results:")
    print(f"   Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"   Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}")
    print(f"   Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"   Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'ml/models/best_model.pt')
        print(f"   ✅ Saved best model (F1: {val_f1:.4f})")

print("\n🎉 Training complete!")

# Final evaluation on validation set
print("\n📈 Final Validation Report:")
print(classification_report(val_labels, val_preds, target_names=['Legit', 'Fraud']))

# Save training history
history = {
    'best_val_f1': best_val_f1,
    'num_epochs': num_epochs,
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset)
}

with open('ml/models/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n💾 Model saved to: ml/models/best_model.pt")