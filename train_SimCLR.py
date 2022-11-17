import numpy as np
import shutil, time, os, requests, random, copy
import gc
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from loss_SimCLR import SimCLR_Loss

from network_SimCLR import PreModel

#Optimizer
learning_rate = 0.01
batch_size = 128
optimizer = optim.Adam(lr=learning_rate)

# "decay the learning rate with the cosine decay schedule without restarts"
#SCHEDULER OR LINEAR EWARMUP
#warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)
#SCHEDULER FOR COSINE DECAY
#mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

#loss function
criterion = SimCLR_Loss(batch_size, temperature = 0.5)

def save_model(model, optimizer, scheduler, current_epoch, name):
    out = os.path.join('/273KELVIN/saved_models/',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)

model = PreModel('resnet50').to('cuda:0')

nr = 0
current_epoch = 0
epochs = 100
train_loss = []
val_loss = []

for e in range(epochs):
        
    print(f"Epoch [{e}/{epochs}]\t")
    stime = time.time()

    model.train()
    train_loss_epoch = 0
    
    
