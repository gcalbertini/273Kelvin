import torchvision.models as models
import torch.nn as nn
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

trained_fastercnn_path = './model_2_batch_4_mom_0.9_decay_0.0001_epochs_11_lr_0.01_backbone_SimCLR_RPN.pt'

def get_model(num_classes=100):

    checkpoint = torch.load(trained_fastercnn_path)

    backbone = get_backbone_fpn()

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) 
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                    aspect_ratios=((0.25, 0.5, 1.0, 2.0),) * len(anchor_sizes))  # (1:2, 1:1, 2:1); 3 AR + 3 scales scales shown to work best in Faster RCNN paper wrt mAP; probably keep samem

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1','2','3'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone=backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)
    
    model.load_state_dict(checkpoint)

    return model

def get_backbone_fpn():

    resnet = models.resnet18(weights=None)
    resnet.maxpool = nn.Identity()
    resnet.fc = nn.Identity()

    returned_layers=[1, 2, 3, 4]
    in_channels_stage2 = resnet.inplanes // 8 # resnet.inplanes=512
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(backbone=resnet, in_channels_list=in_channels_list, return_layers=return_layers, out_channels=256, extra_blocks=extra_blocks)
    
