from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import torch
import warnings

warnings.filterwarnings("ignore")

path = './save_fastrcnn_models/model_3_batch_2_mom_0.9_decay_0.0001_epochs_12_lr_0.005_backbone_SimCLR_RPN.pt'

def get_model(num_classes=100):

    checkpoint = torch.load(path)

    backbone = resnet_fpn_backbone('resnet18', weights=None)

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) 
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                    aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7, sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone, num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)
    
    model.load_state_dict(checkpoint)
    
    return model