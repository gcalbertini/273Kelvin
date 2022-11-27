import torch

from fastrcnn import get_model
from labeled_dataloader import labeled_dataloader
from utils import train_one_epoch
from eval import evaluate

def train(backbone="SimCLR", BATCH_SIZE=16, NUM_WORKERS=2, SHUFFLE=True, DATASET_PATH="/labeled/labeled/", EPOCHS=1, LR=0.001, MOM=0.9, DECAY=0.0005, print_freq=200, verbose=True):

    model = get_model(backbone=backbone, num_classes=100) # if you want to train with mobileye backbone, then: get_model(backbone=None)

    _, train_dataloader = labeled_dataloader(BATCH_SIZE, NUM_WORKERS, SHUFFLE, DATASET_PATH, SPLIT="training")
    _, validation_dataloader = labeled_dataloader(BATCH_SIZE, NUM_WORKERS, SHUFFLE, DATASET_PATH, SPLIT="validation")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY)
    # Use a learning rate scheduler: this means that we will decay the learning rate every <step_size> epoch by a factor of <gamma>
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    if verbose:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq)
        lr_scheduler.step()
        evaluate(model, validation_dataloader, device)
    
    torch.save(model.state_dict(), f"/scratch_tmp/$USER/model__mom_{MOM}_decay_{DECAY}_epoch_{epoch}_lr_{LR}_backbone_{backbone}.pt")

    return model

def main():
    train()

if __name__=="__main__":
    main()
