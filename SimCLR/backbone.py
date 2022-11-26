import torch
import torch.nn as nn
from models import PreModel

class Backbone(nn.Module):
    def __init__(self,premodel):
        super().__init__()
        self.premodel = premodel
        for p in self.premodel.parameters():
            p.requires_grad = False

    def forward(self,x):
        out = self.premodel.pretrained(x)
        return out

def load_backbone():
    model = PreModel('resnet50')
    model.load_state_dict(torch.load('./SimCLR.pt')) # model need to be saved on directory first
    backbone = Backbone(model)
    return backbone

## test ##
# batch = torch.rand(16,3,244,244)
# out = backbone(batch) # -> this is what gets fed to FastRCNN 

