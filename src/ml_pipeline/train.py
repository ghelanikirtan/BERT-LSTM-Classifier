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


def train(
    train_loader: torch.utils.data.DataLoader,
    model: BERTLSTMClassifier,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam) -> OrderedDict:
    """ Training pipeline:
    """
    
    loss_meter = AverageMeter() # loss init
    model.train() 
    pbar = tqdm(total=len(train_loader)) # progress bar
    
    for batch in train_loader:
        
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        # 
        optimizer.zero_grad()
        with autocast(device_type=DEVICE_STR):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # update the loss
        loss_meter.update(loss.item(), input_ids.size(0))
        
        # Progress bar:
        postfix = OrderedDict([
            ('loss', loss_meter.avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    
    pbar.close()
    
    
    return OrderedDict([('loss', loss_meter.avg),])