import os, sys
from typing import OrderedDict
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# 
from ml_pipeline.dataset import NewsDataset
from ml_pipeline.loader import load_data
from ml_pipeline.network import BERTLSTMClassifier
from ml_pipeline.train import train
from ml_pipeline.validate import validate
from constants import *


class MLPipeline:
    
    def __init__(self):
        
        self.tokenizer = None
        self.bert_model = None
        self.model = None
        self.label_encoder = LabelEncoder()

        # 
        self.epochs: int = CONFIG.get('epochs', 30)
        self.num_classes: int = None
        
    def invoke(self):
        
        data = load_data()
        data['label'] = self.label_encoder.fit_transform(data['category'])
        
        self.num_classes = len(self.label_encoder.classes_)
        
        
        # Tokenizer:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Splitting data:
        train_df, val_df = train_test_split(data,
                                            test_size=CONFIG.get('validation_percent', 0.2),
                                            stratify=data['label'],
                                            random_state=50)
        
        
        # Dataset:
        train_dataset = NewsDataset(
            train_df['processed_news'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer            
        )
        val_dataset = NewsDataset(
            val_df['processed_news'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer            
        )
        
        # Data Loader:
        train_loader = DataLoader(train_dataset,
                                  batch_size=CONFIG.get('batch_size', 10),
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=CONFIG.get('num_workers', 4),
                                  pin_memory= True if DEVICE_STR == 'cuda' else False 
                                  )
        val_loader = DataLoader(val_dataset,
                                batch_size=CONFIG.get('batch_size', 10),
                                shuffle=False,
                                drop_last=False,
                                num_workers=CONFIG.get('num_workers', 4),
                                pin_memory= True if DEVICE_STR == 'cuda' else False 
                                )
        
        
        from transformers import AutoModel
        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # BERT-LSTM Model 
        self.model = BERTLSTMClassifier(
            self.bert_model,
            hidden_dim=64,
            num_classes=self.num_classes,
            freeze_bert=True
        ).to(DEVICE)
        
        # Optimizer:
        import torch.nn as nn
        optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        
        trigger = 0
        best_val_loss = float('inf')
        
        #* Training Loop:
        # pbar = tqdm(self.epochs)
        for epoch in range(self.epochs):
            print(f"Epoch [{epoch}/{self.epochs}]")
            #* TRAIN:
            train_log : OrderedDict = train(
                train_loader=train_loader,
                model=self.model,
                criterion=criterion,
                optimizer=optimizer
            )
            
            #* VALIDATE [@ every 5 epochs]
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                val_log : OrderedDict = validate(
                    val_loader=val_loader,
                    model=self.model,
                    criterion=criterion)
                print('loss %.4f - val_loss %.4f' % (train_log['loss'], val_log['loss']))
        
            else:
                val_log = OrderedDict([('loss', float('nan'))])

            trigger += 1
            # Model Saving:::
            if val_log is not None and val_log['loss'] < best_val_loss:
                best_val_loss = val_log['loss']
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss
                }, CONFIG.get('artifacts_path', ARTIFACTS_PATH))
                print(f"=> Saved Best Model at Epoch {epoch+1} with val_loss: {best_val_loss:.4f}")
                trigger = 0
                
            # Early Stopping:::
            if trigger > CONFIG.get("early_stopping_patience", 5):
                print("Early Stopping Triggered.")
                break
                
            
            postfix = OrderedDict([
                ('loss', train_log['loss']),
                ('val_loss', val_log['loss']),
                ('val_acc', val_log['acc'])
            ])
            # pbar.set_postfix(postfix)
            # pbar.update(1)
        # pbar.close()
        print('Training Finished.')
    
    
