import torch
import torch.nn as nn
from models import PreModel
from train import backbone_pretraining

class Backbone(nn.Module):
    def __init__(self,premodel):
        super().__init__()
        self.premodel = premodel
        for p in self.premodel.parameters():
            p.requires_grad = False

    def forward(self,x):
        out = self.premodel.pretrained(x)
        return out

def get_backbone(needs_pretraining=False):

    if needs_pretraining:
        backbone_pretraining()
    model = PreModel('resnet50')
    model.load_state_dict(torch.load('./SimCLR.pt')) # model need to be saved on directory first
    backbone = Backbone(model)
    return backbone

if __name__ == "__main__":
    get_backbone(needs_pretraining=True)

## test ##
# batch = torch.rand(16,3,244,244)
# out = backbone(batch) # -> this is what gets fed to FastRCNN 

