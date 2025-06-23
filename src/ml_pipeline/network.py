from turtle import forward
import torch
import torch.nn as nn


class BERTLSTMClassifier(nn.Module):
    
    def __init__(self, bert_model, hidden_dim, num_classes, freeze_bert=True):
        super(BERTLSTMClassifier, self).__init__()
        self.bert = bert_model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True 
        )
        
        self.fc_layer = nn.Linear(
            in_features=hidden_dim*2,
            out_features=num_classes
        )
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        x = outputs.last_hidden_state
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:,-1,:]
        logits = self.fc_layer(out)
        
        return logits
        
