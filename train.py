#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    LongformerTokenizer, 
    LongformerModel,
    AutoConfig,
    Trainer, 
    TrainingArguments
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

laws_df = pd.read_csv('in.csv')
scraped_content = pd.read_feather('scraped_content.feather')

print(f"wowowow loaded {len(scraped_content)} entries") 

content_dict = {row['name']: row['content'] for _, row in scraped_content.iterrows()}
matched_data = []

for _, law in laws_df.iterrows():
    law_name = law['name']
    if law_name in content_dict:
        matched_data.append({
            'content': content_dict[law_name],
            'police_impact': float(law['police']),
            'drug_impact': float(law['drug']),
            'name': law['name']
        })

matched_df = pd.DataFrame(matched_data)
police_dist = matched_df['police_impact'].value_counts().sort_index()
drug_dist = matched_df['drug_impact'].value_counts().sort_index()
print(f"Police distribution: {police_dist.to_dict()}")
print(f"Drug distribution: {drug_dist.to_dict()}")
police_unique = np.array([-1.0, 0.0, 1.0])
drug_unique = np.array([-1.0, 0.0, 1.0])

police_weights_array = compute_class_weight(
    'balanced', 
    classes=police_unique, 
    y=matched_df['police_impact'].values
)
drug_weights_array = compute_class_weight(
    'balanced', 
    classes=drug_unique, 
    y=matched_df['drug_impact'].values
)

police_weight_dict = dict(zip(police_unique, police_weights_array))
drug_weight_dict = dict(zip(drug_unique, drug_weights_array))

class LongformerForMultiLabelRegression(nn.Module):
    def __init__(self, model_name="allenai/longformer-base-4096"):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(768)
        self.police_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.drug_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, labels=None):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        sequence_output = outputs.last_hidden_state[:, 0] 
        sequence_output = self.dropout(sequence_output)
        if sequence_output.size(0) > 1:
            sequence_output = self.batch_norm(sequence_output)
        
        police_pred = self.police_head(sequence_output).squeeze(-1)
        drug_pred = self.drug_head(sequence_output).squeeze(-1)

        if labels is not None:
            police_labels = labels[:, 0]
            drug_labels = labels[:, 1]
            def smooth_labels(targets, smoothing=0.05):
                return targets * (1.0 - smoothing)
            
            police_labels_smooth = smooth_labels(police_labels)
            drug_labels_smooth = smooth_labels(drug_labels)
            police_loss = nn.MSELoss()(police_pred, police_labels_smooth)
            drug_loss = nn.MSELoss()(drug_pred, drug_labels_smooth)
            
            total_loss = police_loss + drug_loss
            
            return {
                'loss': total_loss,
                'police_predictions': police_pred,
                'drug_predictions': drug_pred
            }
        
        return {
            'police_predictions': police_pred,
            'drug_predictions': drug_pred
        }
        
        
# wow oop in python!!
class LawDataset(Dataset):
    def __init__(self, texts, police_labels, drug_labels, tokenizer, max_length=4096):
        self.texts = texts
        self.police_labels = police_labels
        self.drug_labels = drug_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        global_attention_mask = torch.zeros_like(encoding['input_ids'])
        global_attention_mask[:, 0] = 1  # Set global attention on [CLS] token
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'global_attention_mask': global_attention_mask.flatten(),
            'labels': torch.tensor([self.police_labels[idx], self.drug_labels[idx]], dtype=torch.float)
        }

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerForMultiLabelRegression()

texts = matched_df['content'].values
police_labels = matched_df['police_impact'].values
drug_labels = matched_df['drug_impact'].values

X_train, X_val, y_police_train, y_police_val, y_drug_train, y_drug_val = train_test_split(
    texts, police_labels, drug_labels, test_size=0.2, random_state=42, stratify=police_labels
)

train_dataset = LawDataset(X_train, y_police_train, y_drug_train, tokenizer)
val_dataset = LawDataset(X_val, y_police_val, y_drug_val, tokenizer)

class MultiOutputTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./improved_multi_output_model',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    fp16=True,
    warmup_steps=20,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    learning_rate=5e-7,
    save_total_limit=1,
    dataloader_num_workers=0,     
)


trainer = MultiOutputTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
import os
os.makedirs('./output_model', exist_ok=True)

torch.save(model.state_dict(), './output_model/pytorch_model.bin')

tokenizer.save_pretrained('./output_model')

config = {
    "model_type": "longformer_multi_output",
    "hidden_size": 768,
    "num_labels": 2,
    "max_position_embeddings": 4098
}

import json
with open('./output_model/config.json', 'w') as f:
    json.dump(config, f)
