# Save this as generate_data.py and run it
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

fake = Faker()
np.random.seed(42)

n_reviews = 50_000
fraud_rate = 0.05

data = []
start = datetime(2024, 1, 1)

print("Generating 50,000 reviews... This will take a minute...")

for i in range(n_reviews):
    if i % 10000 == 0:
        print(f"Generated {i} reviews...")
    
    user_id = random.randint(1, 10_000)
    post_date = start + timedelta(days=random.randint(0, 365))
    velocity = np.random.poisson(2)
    is_fraud = random.random() < fraud_rate
    
    if is_fraud:
        text = fake.sentence(nb_words=15, ext_word_list=["amazing", "best ever", "5 stars"]) * random.randint(2, 4)
        rating = 5
        velocity = np.random.poisson(8)
    else:
        text = fake.paragraph(nb_sentences=3)
        rating = random.randint(1, 5)
    
    data.append({
        'review_id': i + 1,
        'user_id': user_id,
        'text': text,
        'rating': rating,
        'velocity': velocity,
        'timestamp': post_date.isoformat(),
        'is_fraud': is_fraud
    })

df = pd.DataFrame(data)
df.to_csv('reviews_dataset.csv', index=False)
print(f"\n✅ Generated {n_reviews} reviews")
print(f"   Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"   Legitimate: {(~df['is_fraud']).sum()} ({(~df['is_fraud']).mean()*100:.1f}%)")
print("\n📁 Saved to reviews_dataset.csv")