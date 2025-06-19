import os
import numpy as np
from typing import Dict, Any
import torch
from torch.utils.data import Dataset



class NewsDataset(Dataset):
    
    def __init__(self, 
                 texts, 
                 labels, 
                 tokenizer, 
                 max_len=64):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encoding text...
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length = self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids' : encoding['input_ids'].squeeze(0),
            'attention_mask' : encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label,dtype=torch.long)
        }
        

