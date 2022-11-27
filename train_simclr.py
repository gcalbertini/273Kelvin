import time
import torch

from unlabeled_dataloader import unlabeled_dataloader
from loss import SimCLR_Loss
from models import PreModel
from lars import LARS

def train(device, dataset, train_loader, model, criterion, optimizer, epochs):

    ### START TRAINING LOOP ###
    nr = 0
    current_epoch = 0
    tr_loss = []
    val_loss = []

    for epoch in range(epochs):
            
        print(f"Epoch [{epoch}/{epochs}]\t")
        stime = time.time()
        model.train()
        tr_loss_epoch = 0
        
        for step, (x_i, x_j) in enumerate(train_loader):

            optimizer.zero_grad()

            x_i = x_i.to(device)
            x_j = x_j.to(device)

            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step(epoch)
            
            if nr == 0 and step % 5 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
            
            tr_loss_epoch += loss.item()
    
        if nr == 0:
            tr_loss.append(tr_loss_epoch / len(dataset))
            print(f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(dataset)}\t")
            current_epoch += 1
        
        time_taken = (time.time()-stime)/60
        print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")

    # saves model
    torch.save(model.state_dict(), "./SimCLR.pt")


def backbone_pretraining(device, DATASET_PATH="./unlabeled_data/", BATCH_SIZE=16, TEMPERATURE=0.5, NUM_WORKERS=2, SHUFFLE=True, IMAGE_SIZE=112, S=1.0, EPOCHS=20, LR=0.2, MOMENTUM=0.9, WEIGHT_DECAY=1e-6):

    dataset, train_loader = unlabeled_dataloader(BATCH_SIZE, NUM_WORKERS, SHUFFLE, DATASET_PATH, IMAGE_SIZE, S)
    model = PreModel('resnet50').to(device)
    criterion = SimCLR_Loss(BATCH_SIZE, TEMPERATURE)
    #optimizer = LARS(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, max_epoch=EPOCHS)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(device, dataset, train_loader, model, criterion, optimizer, EPOCHS)


