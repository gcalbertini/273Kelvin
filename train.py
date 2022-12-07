import torch
from torch.multiprocessing import cpu_count
from tqdm import tqdm
import math
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastrcnn import get_model
from labeled_dataloader import labeled_dataloader
#from utils import train_one_epoch
from eval import evaluate

def train(backbone="SimCLR", BATCH_SIZE=16, EPOCHS=5, NUM_WORKERS=cpu_count()//2, SHUFFLE=False, DATASET_PATH="/labeled/labeled", LR=0.01, MOM=0.9, DECAY=1e-4, print_freq=10, verbose=False):

    model = get_model(backbone=backbone, num_classes=100) # if you want to train with mobileye backbone, then: get_model(backbone=None)

    _, train_dataloader = labeled_dataloader(BATCH_SIZE, NUM_WORKERS, SHUFFLE, DATASET_PATH, SPLIT="training")
    _, validation_dataloader = labeled_dataloader(BATCH_SIZE, NUM_WORKERS, SHUFFLE, DATASET_PATH, SPLIT="validation")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY, nesterov=True)

    # Use a learning rate scheduler: this means that we will decay the learning rate every <step_size> epoch by a factor of <gamma>
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    print("!!! FastRCNN Training START !!!")
    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch)
    evaluate(model, validation_dataloader, device)

    torch.save(model.state_dict(), f"./model__mom_{MOM}_decay_{DECAY}_epochs_{EPOCHS}_lr_{LR}_backbone_{backbone}.pt")

    return model

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

def main():
    train()

if __name__=="__main__":
    main()