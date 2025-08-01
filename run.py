#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import LongformerTokenizer, LongformerModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file

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
    def forward(self, input_ids, attention_mask=None, global_attention_mask=None):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        sequence_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        sequence_output = self.dropout(sequence_output)
        
        # Only apply batch norm if batch size > 1
        if sequence_output.size(0) > 1:
            sequence_output = self.batch_norm(sequence_output)
        
        police_pred = self.police_head(sequence_output).squeeze(-1)
        drug_pred = self.drug_head(sequence_output).squeeze(-1)
        
        return {
            'police_predictions': police_pred,
            'drug_predictions': drug_pred
        }


tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

model = LongformerForMultiLabelRegression()
model_name = "model1"  # Updated to use Longformer model path

model_weights_path = f"./{model_name}/pytorch_model.bin"  # Updated to use .bin format
state_dict = torch.load(model_weights_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

scraped_content = pd.read_feather('scraped_content.feather')


def predict_with_improved_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)
    
    # Create global attention mask for [CLS] token (Longformer requirement)
    global_attention_mask = torch.zeros_like(inputs['input_ids'])
    global_attention_mask[:, 0] = 1  # Set global attention on [CLS] token
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            global_attention_mask=global_attention_mask
        )
        police_pred = outputs['police_predictions'].item()
        drug_pred = outputs['drug_predictions'].item()
    
    return police_pred, drug_pred

def parse_to_discrete_label(continuous_value, threshold_low=-0.05, threshold_high=0.05):
    if continuous_value < threshold_low:
        return -1
    elif continuous_value > threshold_high:
        return 1
    else:
        return 0

all_predictions = []

for idx, row in tqdm(scraped_content.iterrows(), total=len(scraped_content)):
    content = str(row['content'])
    
    try:
        police_pred, drug_pred = predict_with_improved_model(content)
        police_label = parse_to_discrete_label(police_pred)
        drug_label = parse_to_discrete_label(drug_pred)
            
        all_predictions.append({
            'name': row['name'],
            'predicted_police_impact': float(police_pred),
            'predicted_drug_impact': float(drug_pred),
            'police_label': police_label,
            'drug_label': drug_label,
            'content_length': len(content)
        })
        
    except Exception as e:
        all_predictions.append({
            'name': row['name'],
            'predicted_police_impact': 0.0,
            'predicted_drug_impact': 0.0,
            'police_label': 0,
            'drug_label': 0,
            'content_length': len(str(row['content']))
        })

results_df = pd.DataFrame(all_predictions)


output_file = f'law_predictions_longformer_{model_name}.csv'
results_df.to_csv(output_file, index=False)
