import argparse
import builtins
import csv
import os
import random
import socket
import warnings
import torch
import time
import sys
from fastrcnn import get_model
from labeled_dataloader import labeled_dataloader
from eval import evaluate as TA_EVAL
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed as DDP
import torchvision.models as models
from torch.multiprocessing import cpu_count
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import torch.backends.cudnn as cudnn
import torchvision
from utils import *
from socket import gethostname
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='PyTorch SimCLR Bakcbone Training')

# NOTE The store_true option automatically creates a default value of False. Likewise, store_false will default to True when the command-line argument is not present.

# Verbose_off action is store_false
# Example python main.py --verbose_off --> Will not be verbose
# Example python main.py               --> Will be verbose

# Train_backbone action is store_true
# Example python main.py                   --> Will not train backbone
# Example python main.py --train_backbone  --> Trains backbone

# NOTE BATCH SIZES SHOULD BE A MULTIPLE OF GPUs USED AND GREATER THAN THE NUMBER OF GPUs. THE EFFECTIVE BATCH SIZE IS BATCH_SIZE_SPECIFIED*NUM_GPUS*GRAD_ACCUM_STEPS == BATCH_SIZE.
# EFFICIENCY IS DEPENDENT ON GPU HARDWARE ARCHITECTURE.

parser = argparse.ArgumentParser()
parser.add_argument('--path_lbl', default="labeled_data/", metavar='DATA_PATH_LBL', type=str,
                    help="Default path for labeled data; default used is '/labeled/labeled'; note toy set's is 'labeled_data/'")
parser.add_argument('--path_unlbl', default="unlabeled_data/", metavar='DATA_PATH_UNLBL', type=str,
                    help="Default path for unlabeled data; default used is'/unlabeled/unlabeled'; note toy set's is 'unlabeled_data/'")
parser.add_argument('--shuffle', action='store_true',
                    dest='SHUFFLE', help="Shuffle data toggle")
parser.add_argument('-voff', '--verbose_off', action='store_false',
                    dest='VERBOSE_OFF', help="Verbose mode toggle")
parser.add_argument('-c', '--classes', default=100, type=int,
                    metavar='NUM_CLASSES', help='Number of classes; default is 100')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='Pretrained (SimCLR) model weights come from this architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-teval', '--ta-evaluate', dest='EVALUATE', action='store_true',
                    help='evaluate model on validation set using course code -- FINAL SUBMISSIONS EVAL CODE!')
parser.add_argument('-opt', '--optimizer', default='Adam',
                    type=str, help='Adam (default) or SGD optimizer')
parser.add_argument('--start_epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-tb', '--tensorboard',
                    action='store_true', help='Tensorboard displays')
parser.add_argument('--save_every', metavar='SAVE_EVERY_EPOCH',
                    default=1, help='Frequency of saving model (per epoch)')

# =====================SIMCLR ONLY: EDIT THESE FOR BACKBONE TRAIN RUN===============================================================================
# NOTE these will come into play after --train_backbone is specified and doing something like: python train.py --train_backbone -bbe 5 -bbbs 12345 --backbone_lr 1e-4
parser.add_argument('-bbe', '--backbone_epochs', default=2,
                    metavar='BACKBONE_EPOCHS', type=int, help="Default number of backbone epochs")
parser.add_argument('-bbcoff', '--backbone_cuda_off',
                    action='store_false', help="Toggle CUDA for backbone training")
parser.add_argument('-bbsd', '--backbone_seed', default=77777,
                    metavar='BACKBONE_SEED', type=int, help="Backbone seed for reproducibility")
parser.add_argument('-bbimg', '--backbone_img_size', default=224,
                    metavar='BACKBONE_IMG_SIZE', type=int, help="Backbone img size")
parser.add_argument('-bbsv', '--backbone_save_directory', default='saved_models/',
                    metavar='BACKBONE_SAVE_DIR_PATH', type=str, help="Backbone save checkpoint directory path")
parser.add_argument('-bblp', '--backbone_load_pretrained',
                    action='store_true', help="Backbone load pretraining")
parser.add_argument('-bbg', '--backbone_grad_accumulate_steps', type=int, default=5,
                    metavar='BACKBONE_GRAD_ACCUM_STEPS', help="Backbone gradient accumulation steps")
parser.add_argument('-bbbs', '--backbone_batch_size', default=96, type=int,
                    metavar='BACKBONE_BATCH_SIZE', help='Backbone batch size to use')
parser.add_argument('-bbemb', '--backbone_embedding_size', type=int, default=128,
                    metavar='BACKBONE_EMBED_SIZE', help='Backbone embedding size')
parser.add_argument('-bblr', '--backbone_lr', type=float, default=0.001,
                    metavar='BACKBONE_ADAM_LEARN_RATE', help='Backbone learning rate for ADAM')
parser.add_argument('-bbdk', '--backbone_weight_decay', type=float, default=1e-6,
                    metavar='BACKBONE_WEIGHT_DECAY', help='Backbone weight decay')
parser.add_argument('-bbtmp', '--backbone_temperature', type=float, default=0.1,
                    metavar='BACKBONE_TEMP', help='Backbone temperature parameter (set to 0.1 or 0.5)')
parser.add_argument('-bbcp', '--backbone_checkpoint_path', default='./SimCLR_ResNet18.ckpt',
                    metavar='BACKBONE_CHECKPOINT_PATH', type=str, help="Backbone checkpoint path")
parser.add_argument('-bbr', '--backbone_resume', action='store_true',
                    help="Backbone resume training from checkpoint; default is False")

def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        # torch.backends.cudnn.enabled = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # WARNING: create model - already does distributed GPU train from lightning code so avoid interfering with custom DDP process below...
    print(f'Retrieving backbone model...')
    model = get_model(args, backbone=args.backbone, num_classes=args.classes)
    print('Done.')

    # Print model's state_dict
    if not args.verbose:
        print("Backbones's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    # Save model weights
    torch.save(model.state_dict(), './backbone')

    if args.ta_evaluate:
        # TA code entry point
        # TODO Consider making validation-set-specific batch size?
        _, val_loader = labeled_dataloader(args.batch_size, int(
            os.environ["SLURM_CPUS_PER_TASK"]), args.shuffle, args.path_lbl, SPLIT="validation")
        TA_EVAL(model, val_loader, torch.device('cuda:0')
                if torch.cuda.is_available() else torch.device('cpu'))
        return


if __name__ == "__main__":
    main()
