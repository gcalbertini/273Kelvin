import torch
import torch.nn as nn
from train_simclr import backbone_pretraining
from models import PreModel

class Backbone(nn.Module):
    def __init__(self,premodel):
        super().__init__()
        self.out_channels = 0
        self.premodel = premodel
        for p in self.premodel.parameters():
            p.requires_grad = False

    def forward(self,x):
        out = self.premodel.pretrained(x)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        return out

def get_backbone(train=False, **kwargs):
    """
    DEFAULT VALUES:
    DATASET_PATH="./unlabeled_data/", BATCH_SIZE=16, TEMPERATURE=0.5, NUM_WORKERS=2, SHUFFLE=True
    IMAGE_SIZE=112, S=1.0, EPOCHS=20, LR=0.2, MOMENTUM=0.9, WEIGHT_DECAY=1e-6

    TEMPERATURE: hyperparameter used in the SimCLR loss
    S: is color distortion for transformation of unlabeled images
    LR: LARS learning rate
    MOMENTUM: LARS momentum
    WEIGHT_DECAY: LARS weight decay

    To change default values, you need to pass them to get_backbone() with needs_pretraining=True and train a new backbone, for e.g.:
    get_backbone(needs_pretraining=True, BATCH_SIZE=8, IMAGE_SIZE=224):
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)

    if train:
        backbone_pretraining(device, **kwargs)
        
    #model = PreModel('resnet50')
    #model.load_state_dict(torch.load("/scratch_tmp/$USER/SimCLR_0.pt", map_location=device)) # model need to be saved on directory first
    #backbone = Backbone(model)
    #return backbone

def main():
    model = get_backbone(train=True, NUM_WORKERS=12, EPOCHS=1, DATASET_PATH="/unlabeled/unlabeled", SHUFFLE=False, IMAGE_SIZE=224, BATCH_SIZE=2)

if __name__=="__main__":
    main()
