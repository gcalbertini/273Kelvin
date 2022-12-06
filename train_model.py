import argparse
import csv
import os
import random
import socket
import warnings
import torch
import time
import sys
from labeled_dataloader import labeled_dataloader
from eval import evaluate as TA_EVAL
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed as DDP
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision
from utils import *
from socket import gethostname
from backbone import Backbone
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# REF https://debuggercafe.com/a-simple-pipeline-to-train-pytorch-faster-rcnn-object-detection-model/
# REF https://github.dev/tczhangzhi/pytorch-distributed/blob/cd12856420858b14e02873e7d5c8cc7bb5aab5b0/distributed_slurm_main.py#L290
# REF https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
# REF https://pytorch.org/docs/stable/distributed.html
# REF https://github.dev/lkskstlr/distributed_data_parallel_slurm_setup
# REF https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
# REF REF https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch FasterRCNN Training')
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
# NOTE: Ensure this matches backbone arch of choice
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='Pretrained (SimCLR) model weights come from this architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--path_lbl', default="labeled_data/", metavar='DATA_PATH_LBL', type=str,
                    help="Default path for labeled data; default used is '/labeled/labeled'; note toy set's is 'labeled_data/'")
parser.add_argument('--path_unlbl', default="unlabeled_data/", metavar='DATA_PATH_UNLBL', type=str,
                    help="Default path for unlabeled data; default used is'/unlabeled/unlabeled'; note toy set's is 'unlabeled_data/'")
parser.add_argument('--shuffle', action='store_true',
                    dest='SHUFFLE', help="Shuffle data toggle")
parser.add_argument('-voff', '--verbose_off', action='store_false',
                    dest='VERBOSE_OFF', help="Verbose mode toggle")
parser.add_argument('-teval', '--ta_evaluate', dest='EVALUATE', action='store_true',
                    help='evaluate model on validation set using course code -- FINAL SUBMISSIONS EVAL CODE!')
parser.add_argument('-opt', '--optimizer', default='Adam',
                    type=str, help='Adam (default) or SGD optimizer')
parser.add_argument('--start_epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-tb', '--tensorboard',
                    action='store_true', help='Tensorboard displays')
parser.add_argument('--save_every', metavar='SAVE_EVERY_EPOCH',
                    default=1, help='Frequency of saving model (per epoch)')
parser.add_argument('-bb', '--backbone', default="SimCLR", type=str, metavar='BACKBONE',
                    help="Backbone to use; default is SimCLR. Set to 'None' for mobilenet_v2.")

# =====================FASTERCNN-SIMCLR: EDIT THESE FOR FULL MODEL RUN AFTER BACKBONE TRAIN===============================================================
parser.add_argument('-bs', '--batch_size', default=64,
                    type=int, metavar='BATCH_SIZE', help='Batch size to use')
parser.add_argument('-e', '--epochs', default=5, metavar='EPOCHS',
                    type=int, help="Default number of epochs")
parser.add_argument('-lr', '--learn_rate', default=0.001,
                    metavar='LEARN_RATE', type=float, help="Default learning rate")
parser.add_argument('-mom', '--momentum', default=0.9,
                    metavar='MOMENTUM', type=float, help="Default momentum")
parser.add_argument('-dk', '--weight_decay', default=0.0005,
                    metavar='WEIGHT_DECAY', type=float, help="Default weight decay")
parser.add_argument('-p', '--print_freq', default=200,
                    metavar='PRINT_FREQ', type=int, help="Default print frequency")
parser.add_argument('-s', '--step', default=50, metavar='SCHEDULER_STEP',
                    type=int, help="Default step size for scheduler")
parser.add_argument('-g', '--gamma', default=0.2, metavar='SCHEDULER_GAMMA',
                    type=float, help="Default gamma factor for scheduler")
parser.add_argument('--pretrained', dest='pretrained',
                    action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-f', '--freeze', action='store_true', help='Freeze backbone weights; default is False')


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
        total_loss += losses.item()
        total_correct += get_num_correct(preds, label_target)

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


def setup(rank, world_size):
    # Find free port - type 'netstat -lntu' to see free TCP ports
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    print(f'MASTER PORT={port}')
    os.environ['MASTER_PORT'] = str(port)
    s.close()

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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

    args.num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    print(args.num_workers)
    print(world_size)
    print(rank)
    print(gpus_per_node)
    print(torch.cuda.device_count())

    assert gpus_per_node == len([torch.cuda.device(i)
                                for i in range(torch.cuda.device_count())])
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are"
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    print(
        'Loading in backbone to rank {rank} of {world_size} on {gethostname()}...', flush=True)
    path = './' + str(args.arch)+ '_backbone_weights.ckpt'
    # load your model architecture/module
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Identity()
    model.load_state_dict(torch.load('./resnet18_backbone_weights.ckpt')['model_state_dict'])
    print('Done.', flush=True)

    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False

    setup(rank, world_size)
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    # Data loading code
    print(f'Loading data to rank {rank} of {world_size} on {gethostname()}...', flush=True)
    train_dataset, train_dataloader = labeled_dataloader(args.batch_size, args.num_workers, args.shuffle, args.path_lbl, SPLIT="training")
    print('Done.', flush=True)

    # TODO Consider making validation-set-specific batch size?
    _, val_loader = labeled_dataloader(args.batch_size, int(
        os.environ["SLURM_CPUS_PER_TASK"]), args.shuffle, args.path_lbl, SPLIT="validation")

    # Now make dataloader for DDP
    print(
        f'Generating DDP loader for rank {rank} of {world_size} on {gethostname()}...', flush=True)
    train_dataloader = DDP.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    print('Done.', flush=True)

    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

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
        print('Dumbass.', flush=True)
        sys.exit()

    scheduler = StepLR(args.optimizer, step_size=args.step, gamma=args.gamma)

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    cudnn.benchmark = True

    log_csv = "distributed.csv"

    if rank == 0:
        tb = None
        if args.tensorboard:
            print(
                f'Creating Tensorboard summary writer to rank {rank} of {world_size} on {gethostname()} and adding data...', flush=True)
            tb = SummaryWriter()
            _, train_dataloader = labeled_dataloader(args.batch_size, int(
                os.environ["SLURM_CPUS_PER_TASK"]), args.shuffle, args.path_lbl, SPLIT="training")
            images, _ = next(iter(train_dataloader))
            grid = torchvision.utils.make_grid(images)
            tb.add_image("images", grid)
            tb.add_graph(model, images)
            print('Done.', flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()

        train(train_dataloader, ddp_model,
              optimizer, epoch, local_rank, args, tb)

        if rank == 0:
            # evaluate on validation set
            acc1 = validate(val_loader, ddp_model, local_rank, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        scheduler.step()

        epoch_end = time.time()

        if rank == 0:
            with open(log_csv, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [
                    time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(epoch_start)),
                    epoch_end - epoch_start
                ]
                csv_write.writerow(data_row)

            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': ddp_model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)

            if epoch % args.save_every == 0:
                torch.save(ddp_model.state_dict(), "fasterRCNN_SimCLR.pt")

    dist.destroy_process_group()

    if args.tensorboard:
        for name, weight in ddp_model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)
            tb.close()

    if args.ta_evaluate:
        # TA code entry point
        TA_EVAL(ddp_model, val_loader, torch.device('cuda:0')
                if torch.cuda.is_available() else torch.device('cpu'))
        return


if __name__ == "__main__":
    main()
