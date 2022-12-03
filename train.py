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
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore", category=PossibleUserWarning)

# REF https://debuggercafe.com/a-simple-pipeline-to-train-pytorch-faster-rcnn-object-detection-model/
# REF https://github.dev/tczhangzhi/pytorch-distributed/blob/cd12856420858b14e02873e7d5c8cc7bb5aab5b0/distributed_slurm_main.py#L290
# REF https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
# REF https://pytorch.org/docs/stable/distributed.html
# REF https://github.dev/lkskstlr/distributed_data_parallel_slurm_setup
# REF https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch FasterRCNN Training')
#NOTE The store_true option automatically creates a default value of False. Likewise, store_false will default to True when the command-line argument is not present.

    # Verbose_off action is store_false
    # Example python main.py --verbose_off --> Will not be verbose
    # Example python main.py               --> Will be verbose

    # Train_backbone action is store_true
    # Example python main.py                   --> Will not train backbone
    # Example python main.py --train_backbone  --> Trains backbone

    #NOTE BATCH SIZES SHOULD BE A MULTIPLE OF GPUs USED AND GREATER THAN THE NUMBER OF GPUs. THE EFFECTIVE BATCH SIZE IS BATCH_SIZE_SPECIFIED*NUM_GPUS*GRAD_ACCUM_STEPS == BATCH_SIZE.
    # EFFICIENCY IS DEPENDENT ON GPU HARDWARE ARCHITECTURE. 
parser = argparse.ArgumentParser()
parser.add_argument('--path_lbl', default="labeled_data/", metavar='DATA_PATH_LBL', type=str, help="Default path for labeled data; default used is '/labeled/labeled'; note toy set's is 'labeled_data/'")
parser.add_argument('--path_unlbl', default="unlabeled_data/", metavar='DATA_PATH_UNLBL', type=str, help="Default path for unlabeled data; default used is'/unlabeled/unlabeled'; note toy set's is 'unlabeled_data/'")
parser.add_argument('--shuffle', action='store_true', dest='SHUFFLE', help="Shuffle data toggle")
parser.add_argument('-voff', '--verbose_off', action='store_false', dest='VERBOSE_OFF', help="Verbose mode toggle")
parser.add_argument('-c', '--classes', default=100, type=int, metavar='NUM_CLASSES', help='Number of classes; default is 100')
parser.add_argument('-a','--arch',metavar='ARCH',default='resnet18',choices=model_names,
                help='Pretrained (SimCLR) model weights come from this architecture: ' + ' | '.join(model_names) +
                ' (default: resnet18)')
parser.add_argument('-eval','--evaluate',dest='EVALUATE',action='store_true',help='evaluate model on validation set -- FINAL SUBMISSIONS EVAL CODE!')
parser.add_argument('-opt','--optimizer',default='Adam',type=str,help='Adam (default) or SGD optimizer')
parser.add_argument('--start_epoch',default=0,type=int,metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-tb','--tensorboard', action='store_true', help='Tensorboard displays')

# DDP configs:
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend (default nccl or gloo)')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
#=====================FASTERCNN-SIMCLR: EDIT THESE FOR FULL MODEL RUN AFTER BACKBONE TRAIN===============================================================
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
parser.add_argument('-f', '--freeze', action='store_true', dest='FREEZE', help='Freeze backbone weights; default is False')
parser.add_argument('--pretrained',dest='pretrained', action='store_true',help='use pre-trained model')
parser.add_argument('--seed', default=None,type=int,help='seed for initializing training. ')
#=====================SIMCLR ONLY: EDIT THESE FOR BACKBONE TRAIN RUN===============================================================================
# NOTE these will come into play after --train_backbone is specified and doing something like: python train.py --train_backbone -bbe 5 -bbbs 12345 --backbone_lr 1e-4
parser.add_argument('-bbe','--backbone_epochs', default=10, metavar='BACKBONE_EPOCHS', type=int, help="Default number of backbone epochs")
parser.add_argument('-bbcoff','--backbone_cuda_off', action='store_false', help="Toggle CUDA for backbone training")
parser.add_argument('-bbsd','--backbone_seed', default=77777, metavar='BACKBONE_SEED', type=int, help="Backbone seed for reproducibility")
parser.add_argument('-bbimg','--backbone_img_size', default=224, metavar='BACKBONE_IMG_SIZE', type=int, help="Backbone img size")
parser.add_argument('-bbsv','--backbone_save_directory', default='saved_models/', metavar='BACKBONE_SAVE_DIR_PATH', type=str, help="Backbone save checkpoint directory path")
parser.add_argument('-bblp','--backbone_load_pretrained', action='store_true', help="Backbone load pretraining")
parser.add_argument('-bbg','--backbone_grad_accumulate_steps', type=int, default = 5, metavar='BACKBONE_GRAD_ACCUM_STEPS', help="Backbone gradient accumulation steps")
parser.add_argument('-bbbs', '--backbone_batch_size', default=96, type=int, metavar='BACKBONE_BATCH_SIZE', help='Backbone batch size to use')
parser.add_argument('-bbemb','--backbone_embedding_size', type=int, default = 128, metavar='BACKBONE_EMBED_SIZE', help='Backbone embedding size')
parser.add_argument('-bblr','--backbone_lr', type=float, default = 3e-4, metavar='BACKBONE_ADAM_LEARN_RATE', help='Backbone learning rate for ADAM')
parser.add_argument('-bbdk','--backbone_weight_decay', type=float, default = 1e-6, metavar='BACKBONE_WEIGHT_DECAY', help='Backbone weight decay')
parser.add_argument('-bbtmp','--backbone_temperature', type=float, default=0.1, metavar='BACKBONE_TEMP', help='Backbone temperature parameter (set to 0.1 or 0.5)')
parser.add_argument('-bbcp','--backbone_checkpoint_path', default='./SimCLR_ResNet18.ckpt', metavar='BACKBONE_CHECKPOINT_PATH', type=str, help="Backbone checkpoint path")
parser.add_argument('-bbr','--backbone_resume', action='store_true', help="Backbone resume training from checkpoint; default is False")


def train(train_loader, model, optimizer, epoch, gpu, args, tb):

    total_loss = 0
    total_correct = 0

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_id, (images, label_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(gpu, non_blocking=True)
        label_target = label_target.cuda(gpu, non_blocking=True)

        # compute output and simple metrics
        preds = model(images)
        loss_dict = model(images, label_target)
        losses = sum(loss for loss in loss_dict.values())
        total_loss+= losses.item()
        total_correct+= get_num_correct(preds, label_target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(preds, label_target, topk=(1, 5))
        losses.update(losses.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do Adam/SGD step
        optimizer.zero_grad()
        loss_dict.backward()
        optimizer.step()

        if args.tensorboard:
            tb.add_scalar("Loss", total_loss, batch_id)
            tb.add_scalar("Correct", total_correct, epoch)
            tb.add_scalar("Accuracy@1", acc1, epoch)
            tb.add_scalar("Accuracy@5", acc5, epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % args.print_freq == 0:
            progress.display(batch_id)


best_acc1 = 0

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

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
         args.world_size = int(os.environ["SLURM_NPROCS"])

    args.local_rank = int(os.environ["SLURM_PROCID"])
    args.world_size = int(os.environ["SLURM_NPROCS"])
    ngpus_per_node = torch.cuda.device_count()
    

    s=socket.socket()
    s.bind(("", 0))
    args.master_port = int(s.getsockname()[1])
    s.close()
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    print(f'Local rank is {args.local_rank} and world size is {args.world_size}')
    print(f'Using {ngpus_per_node} GPUs per node')

    job_id = os.environ["SLURM_JOBID"]
    print(f'Job: {job_id}')


    '''
    The srun command has two different modes of operation. First, if not run within an existing
    job (i.e. not within a Slurm job allocation created by salloc or sbatch),
    then it will create a job allocation and spawn an application. 
    If run within an existing allocation (as we are doing with the sbatch), the srun command only spawns the application so we 
    must spawn ourselves as below.
    '''
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))



def main_worker(gpu, ngpus_per_node, args):
    tb = SummaryWriter()
    global best_acc1
    rank = args.local_rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl',
                            world_size=args.world_size,
                            rank=rank)

    # create model - already does distributed GPU train from borrowed code
    model = get_model(args, backbone=args.backbone, num_classes=args.classes)
    model_without_ddp = model

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
            
    
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        print('Dumbass.')
        sys.exit()

    cudnn.benchmark = True

    # Data loading code
    train_dataset, train_dataloader = labeled_dataloader(args.batch_size, 4, args.shuffle, args.path_lbl, SPLIT="training")
    _, validation_dataloader = labeled_dataloader(args.batch_size, 4, args.shuffle, args.path_lbl, SPLIT="validation")

    if args.tensorboard:
        images, _ = next(iter(train_dataloader)) 
        grid = torchvision.utils.make_grid(images)
        tb.add_image("images", grid)
        tb.add_graph(model, images)

    train_sampler = DDP.DistributedSampler(train_dataset)

    
    if args.evaluate:
        # TA code entry point
        TA_EVAL(model, validation_dataloader, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        return

    log_csv = "distributed.csv"


    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_dataloader, model, optimizer, epoch, gpu, args, tb)

        # evaluate on validation set
        acc1 = validate(validation_dataloader, model, gpu, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        epoch_end = time.time()

        with open(log_csv, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)),
                epoch_end - epoch_start
            ]
            csv_write.writerow(data_row)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)

    if args.tensorboard:
        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

    tb.close()


if __name__=="__main__":
    main()
