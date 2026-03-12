# ml/export/convert_to_onnx.py
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import time

# Custom Model (same architecture as training)
class FraudDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + 1, 2)
        
    def forward(self, input_ids, attention_mask, velocity):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch, 768)
        
        # Fix: Ensure velocity has correct shape (batch, 1)
        if velocity.dim() == 1:
            velocity = velocity.unsqueeze(1)  # Shape: (batch, 1)
        elif velocity.dim() == 3:
            velocity = velocity.squeeze(1)  # Remove extra dimension if present
            
        combined = torch.cat([cls_embedding, velocity], dim=1)  # Shape: (batch, 769)
        combined = self.dropout(combined)
        output = self.classifier(combined)  # Shape: (batch, 2)
        
        return output

print("🚀 Starting ONNX conversion...")

# Load model
device = torch.device('cpu')
model = FraudDetectionModel().to(device)
model.eval()

# Load trained weights
model_path = 'ml/models/best_model.pt'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded trained model from {model_path}")
else:
    print(f"❌ Model not found at {model_path}")
    print("Please move best_model.pt to ml/models/ folder")
    exit(1)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create dummy input for ONNX export with fixed sequence length
max_length = 128
dummy_text = ["This is a sample review for ONNX export"]
dummy_velocity = torch.tensor([5.0], dtype=torch.float32)  # Shape: (1,)

# Tokenize dummy input with fixed max_length
dummy_encodings = tokenizer(
    dummy_text,
    truncation=True,
    padding='max_length',  # Pad to max_length
    max_length=max_length,
    return_tensors='pt'
)

print(f"Input shapes:")
print(f"  input_ids: {dummy_encodings['input_ids'].shape}")
print(f"  attention_mask: {dummy_encodings['attention_mask'].shape}")
print(f"  velocity: {dummy_velocity.shape}")

# Export to ONNX
print("\n🔄 Converting to ONNX format...")
onnx_path = 'ml/models/model.onnx'

try:
    # Export with fixed sequence length
    torch.onnx.export(
        model,
        (dummy_encodings['input_ids'], 
         dummy_encodings['attention_mask'], 
         dummy_velocity),
        onnx_path,
        input_names=['input_ids', 'attention_mask', 'velocity'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},  # Only batch size is dynamic, sequence length fixed
            'attention_mask': {0: 'batch_size'},
            'velocity': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=14
    )
    print(f"✅ ONNX model saved to {onnx_path}")
except Exception as e:
    print(f"❌ Export failed: {e}")
    exit(1)

# Verify ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model verification passed")

# Quantize to INT8 for faster inference
print("\n🔄 Quantizing model to INT8...")
quantized_path = 'ml/models/model_quantized.onnx'

quantize_dynamic(
    onnx_path,
    quantized_path,
    weight_type=QuantType.QInt8
)

print(f"✅ Quantized model saved to {quantized_path}")

# Test inference speed
print("\n⏱️ Testing inference speed...")
ort_session = onnxruntime.InferenceSession(quantized_path)

# Prepare test input with fixed sequence length
test_texts = [
    "This product is amazing! Best purchase ever!",
    "Worst quality, broke after one use",
    "Good value for money, would recommend"
]

# Test single inference
start_time = time.time()
for text in test_texts:
    # Tokenize with same max_length and padding
    encodings = tokenizer(
        text, 
        truncation=True, 
        padding='max_length',  # Important: use same padding as export
        max_length=max_length, 
        return_tensors='pt'
    )
    
    # Prepare ONNX input
    ort_inputs = {
        'input_ids': encodings['input_ids'].numpy().astype(np.int64),
        'attention_mask': encodings['attention_mask'].numpy().astype(np.int64),
        'velocity': np.array([5.0], dtype=np.float32)
    }
    
    # Run inference
    outputs = ort_session.run(None, ort_inputs)
    
single_time = (time.time() - start_time) / len(test_texts) * 1000  # in ms

# Test batch inference
print("\n📊 Batch size performance:")
batch_sizes = [1, 8, 16, 32, 64]

for batch_size in batch_sizes:
    # Create batch of dummy texts
    batch_texts = ["Sample review"] * batch_size
    
    # Tokenize batch with same max_length and padding
    encodings = tokenizer(
        batch_texts, 
        truncation=True, 
        padding='max_length',  # Important: use same padding as export
        max_length=max_length, 
        return_tensors='pt'
    )
    
    # Prepare ONNX input
    ort_inputs = {
        'input_ids': encodings['input_ids'].numpy().astype(np.int64),
        'attention_mask': encodings['attention_mask'].numpy().astype(np.int64),
        'velocity': np.array([5.0] * batch_size, dtype=np.float32)
    }
    
    # Warmup
    for _ in range(5):
        _ = ort_session.run(None, ort_inputs)
    
    # Measure
    start_time = time.time()
    num_iterations = 50
    for _ in range(num_iterations):
        outputs = ort_session.run(None, ort_inputs)
    
    total_time = (time.time() - start_time) * 1000  # ms
    avg_time = total_time / num_iterations
    
    print(f"   Batch size {batch_size:2d}: {avg_time:.2f} ms average")
    print(f"   Per review: {avg_time/batch_size:.2f} ms")

print(f"\n✅ Single inference: {single_time:.2f} ms")

# Check if requirement is met
if single_time < 200:
    print("✅ Requirement MET: <200ms inference")
else:
    print("⚠️ Warning: Inference time >200ms")

# Compare file sizes
original_size = os.path.getsize(model_path) / (1024*1024)
onnx_size = os.path.getsize(onnx_path) / (1024*1024)
quantized_size = os.path.getsize(quantized_path) / (1024*1024)

print(f"\n📁 File sizes:")
print(f"   PyTorch model: {original_size:.2f} MB")
print(f"   ONNX model: {onnx_size:.2f} MB")
print(f"   Quantized ONNX: {quantized_size:.2f} MB")
print(f"   Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
print("✅ ONNX export complete!")