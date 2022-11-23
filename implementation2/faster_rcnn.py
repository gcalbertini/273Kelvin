# REFERENCE: https://debuggercafe.com/using-any-torchvision-pretrained-model-as-backbone-for-pytorch-faster-rcnn/

import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from utils import *
from eval import evaluate

from labeled_dataloaders import LabeledDataset, collate_fn
def create_model(num_classes):
    # Load the pretrained SimCLR backbone
	# TODO
    backbone = get_simCLR_model().features
    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is ### for SimCLR
	#TODO
    backbone.out_channels = ####
    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    print(model)
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
