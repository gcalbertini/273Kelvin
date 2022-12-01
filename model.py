import torch
import argparse
from fastrcnn import get_model as rcnn_model

def get_model():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_backbone', action='store_true', help='Train backbone toggle')
    parser.add_argument('-o', '--output_size', default=1, type=int, help="Output size for the backbone") #TODO is this 1?? See fastercnn.py
    parser.add_argument('-bb', '--backbone', default="SimCLR", type=str, metavar='BACKBONE', help = "Backbone to use; default is SimCLR. Set to 'None' for mobilenet_v2.")
    parser.add_argument('-bs', '--batch_size', default=64, type=int, metavar='BATCH_SIZE', help='Batch size to use')
    parser.add_argument('-e', '--epochs', default=5, metavar='EPOCHS', type=int, help="Default number of epochs")
    parser.add_argument('-lr', '--learn_rate', default=0.001, metavar='LEARN_RATE', type=float, help="Default learning rate")
    parser.add_argument('-mom', '--momentum', default=0.9, metavar='MOMENTUM', type=float, help="Default momentum")
    parser.add_argument('-dk', '--weight_decay', default=0.0005, metavar='WEIGHT_DECAY', type=float, help="Default weight decay")
    parser.add_argument('-p', '--print_freq', default=200, metavar='PRINT_FREQ', type=int, help="Default print frequency")
    parser.add_argument('-s', '--step', default=50, metavar='SCHEDULER_STEP', type=int, help="Default step size for scheduler")
    parser.add_argument('-g', '--gamma', default=0.2, metavar='SCHEDULER_GAMMA', type=float, help="Default gamma factor for scheduler")
    parser.add_argument('-f', '--freeze', action='store_true', help='Freeze backbone weights; default is False')
    # parser.add_argument('-cp','--checkpoint_path', default='./Fast_RCNN.ckpt', metavar='MODEL_CHECKPOINT_PATH', type=str, help="Full model checkpoint path")
    # parser.add_argument('-r','--resume', action='store_true', help="Model resume training from checkpoint; default is False")
    # parser.add_argument('-sv','--save_directory', default='saved_full_models/', metavar='FULL_MODEL_SAVE_DIR_PATH', type=str, help="Full model save checkpoint directory path")
    # parser.add_argument('-lp','--load_pretrained', action='store_true', help="Model load pretraining")

    #=====================SIMCLR ONLY: EDIT THESE FOR BACKBONE TRAIN RUN===============================================================================
    # NOTE these will come into play after --train_backbone is specified and doing something like: python train.py --train_backbone -bbe 5 -bbbs 12345 --backbone_lr 1e-4

    parser.add_argument('-bbe','--backbone_epochs', default=10, metavar='BACKBONE_EPOCHS', type=int, help="Default number of backbone epochs")
    parser.add_argument('-bbcoff','--backbone_cuda_off', action='store_false', help="Toggle CUDA for backbone training")
    parser.add_argument('-bbsd','--backbone_seed', default=77777, metavar='BACKBONE_SEED', type=int, help="Backbone seed for reproducibility")
    parser.add_argument('-bbimg','--backbone_img_size', default=224, metavar='BACKBONE_IMG_SIZE', type=int, help="Backbone img size")
    parser.add_argument('-bbsv','--backbone_save_directory', default='saved_backbone_models/', metavar='BACKBONE_SAVE_DIR_PATH', type=str, help="Backbone save checkpoint directory path")
    parser.add_argument('-bblp','--backbone_load_pretrained', action='store_true', help="Backbone load pretraining")
    parser.add_argument('-bbg','--backbone_grad_accumulate_steps', type=int, default = 5, metavar='BACKBONE_GRAD_ACCUM_STEPS', help="Backbone gradient accumulation steps")
    parser.add_argument('-bbbs', '--backbone_batch_size', default=96, type=int, metavar='BACKBONE_BATCH_SIZE', help='Backbone batch size to use')
    parser.add_argument('-bbemb','--backbone_embedding_size', type=int, default = 128, metavar='BACKBONE_EMBED_SIZE', help='Backbone embedding size')
    parser.add_argument('-bblr','--backbone_lr', type=float, default = 3e-4, metavar='BACKBONE_ADAM_LEARN_RATE', help='Backbone learning rate for ADAM')
    parser.add_argument('-bbdk','--backbone_weight_decay', type=float, default = 1e-6, metavar='BACKBONE_WEIGHT_DECAY', help='Backbone weight decay')
    parser.add_argument('-bbtmp','--backbone_temperature', type=float, default=0.1, metavar='BACKBONE_TEMP', help='Backbone temperature parameter (set to 0.1 or 0.5)')
    parser.add_argument('-bbcp','--backbone_checkpoint_path', default='./SimCLR_ResNet18.ckpt', metavar='BACKBONE_CHECKPOINT_PATH', type=str, help="Backbone checkpoint path")
    parser.add_argument('-bbr','--backbone_resume', action='store_true', help="Backbone resume training from checkpoint; default is False")

    args = parser.parse_args()
    model = rcnn_model(args, backbone=args.backbone, num_classes=100)
    model.load_state_dict(torch.load("./saved_models/model__mom_0.9_decay_0.0005_epochs_5_lr_0.001_backbone_SimCLR.pt"))
    
    return model
