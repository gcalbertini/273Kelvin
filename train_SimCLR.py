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
from unlabeled_dataloaders import main
from network_SimCLR import PreModel

#Hyper parameters
learning_rate = 0.01
batch_size = 128
optimizer = optim.Adam(lr=learning_rate)
current_epoch = 0
epochs = 100

#loss function
criterion = SimCLR_Loss(batch_size, temperature = 0.5)

#model
model_SimCLR = PreModel('resnet50').to('cuda:0')

#helper function
# def save_model(model, optimizer, scheduler, current_epoch, name):
#     out = os.path.join('/273KELVIN/saved_models/',name.format(current_epoch))

#     torch.save({'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict':scheduler.state_dict()}, out)

train_loader = main()

def train(model):
    train_loss = []

    for e in range(epochs):
        print(f"Epoch [{e}/{epochs}]\t")
        stime = time.time()

        model.train()
        train_loss_epoch = 0
    
        for img in train_loader:
            #optimizer.zero_grad()
            x_i = img[0]
            x_j = img[1]
            x_i = x_i.squeeze().to('cuda:0').float()
            x_j = x_j.squeeze().to('cuda:0').float()
        
            # positive pair, with encoding
            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss.append(train_loss_epoch / len(dl))
        print(f"Epoch [{e}/{epochs}]\t Training Loss: {train_loss_epoch / len(train_loader)}\t ")

        time_taken = (time.time()-stime)/60
        print(f"Epoch [{e}/{epochs}]\t Time Taken: {time_taken} minutes")

    #save_model(model, optimizer, mainscheduler, current_epoch, "trained_SimCLR")
    return model