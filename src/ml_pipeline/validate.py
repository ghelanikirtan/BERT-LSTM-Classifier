import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm
from collections import OrderedDict
# 
from ml_pipeline.utils import AverageMeter
from ml_pipeline.network import BERTLSTMClassifier
from constants import DEVICE, DEVICE_STR


def validate(
    val_loader: torch.utils.data.DataLoader,
    model: BERTLSTMClassifier,
    criterion: nn.CrossEntropyLoss) -> OrderedDict:
    """
    VALIDATION PIPELINE
    """
    
    model.eval()
    loss_meter = AverageMeter()
    target=0
    total=0
    
    with torch.no_grad():
        
        pbar = tqdm(total=len(val_loader)) # progress bar
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels) 

            loss_meter.update(loss.item(), input_ids.size(0))
            preds = torch.argmax(outputs, dim=1)
            target += (preds == labels).sum().item()
            total += labels.size(0) 
            
            # progress bar
            postfix = OrderedDict([
                ('val_loss', loss_meter.avg),
                ('val_acc', target / total )
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    accuracy = target / total
        
    return OrderedDict([('loss', loss_meter.avg),
                        ('acc', accuracy)])
            
            