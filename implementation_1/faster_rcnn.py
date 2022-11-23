# REFERENCE: https://github.com/phil-bergmann/tracking_wo_bnw/tree/master/src/obj_det
import warnings
import torch
from implementation_1.utils import collate_fn, train_one_epoch
from eval import evaluate
from labeled_dataloaders import *
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops import FrozenBatchNorm2d

def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.
    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers


def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    """
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and NUM, with NUM meaning all backbone layers are trainable.
    """
        # TODO
        # so no one seems to understand the SimCLR model layer sequence as is; I propose use the implementation
        # from SimCLR tutorial with pytorch lightning ASAP as progress stalling

    trainable_backbone_layers = _validate_trainable_layers(pretrained or pretrained_backbone, trainable_backbone_layers, NUM???, 3) 

    print(f'trainable_backbone_layers: {trainable_backbone_layers}')

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = 0 # TODO load_state_dict_from_simclr()

    model.load_state_dict(state_dict)
    overwrite_eps(model, 0.0)

    return model

def train_FasterRCNN(model):
    # REFERENCE: https://github.com/pytorch/vision/tree/eb7a0f40ca7a7e269e893c1a8ab5845085c8b219/references/detection
    # https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=DBIoe_tHTQgV

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # FIXME Our dataset has 100 classes or does it have 101 for "background" as tutorial mentions? I think 100 since eval.py specifically listed 100 and never an N/A or Other class
    num_classes = 100

    # FIXME Define paths use our dataset and defined transformations -- establish where this will be
    VALID_DATASET_PATH = '/scratch_tmp/ga947/labeled.sqsh'

    train_dataset = LabeledDataset(
        root=VALID_DATASET_PATH,
        split="training",
        transforms=transform, # albumentations transformation
    )
    validation_dataset =  LabeledDataset(
        root=VALID_DATASET_PATH,
        split="validation",
        transforms=transform, # albumentations transformation
    )
    
    # Now get the dataloaders
    data_loader_train =  torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    data_loader_test =  torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # move model to the right device
    model.to(device)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    LR = 0.0005
    MOM = 0.9
    DECAY = 0.0005
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOM, weight_decay=DECAY)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    #Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # and a learning rate scheduler: this means that we will decay the learning rate every <step_size> epoch by a factor of <gamma>
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # TODO Verify this is compatible with Matteo's DL format
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # Evaluate 
        evaluate(model, data_loader_test, device=device)
        torch.save({model.state_dict()}, f"scratch_tmp/$USER/model__mom_{MOM}_decay_{DECAY}_epoch_{epoch+1}_lr_{LR}.pt")

    return model

