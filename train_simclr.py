import os
import time
import torch

from unlabeled_dataloader import unlabeled_dataloader
from loss import SimCLR_Loss
from models import PreModel
from lars import LARS
import torch.multiprocessing as mp
import torch.distributed as dist


def cleanup():
    dist.destroy_process_group()


def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)

    # this will make all .cuda() calls work properly; You need to use rank instead of the current device (local rank), 
    # otherwise all processes will use the same device if you didn’t set CUDA_VISIBLE_DEVICE or run torch.cuda.set_device(rank)
    torch.cuda.set_device(rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


'''

    local_rank: the rank of the process on the local machine.
    rank: the rank of the process in the network.

 To illustrate that, let;s say you have 2 nodes (machines) with 2 GPU each, you will have a total of 4 processes (p1…p4):

                |    Node1  |   Node2    |
    ____________| p1 |  p2  |  p3  |  p4 |
    local_rank  | 0  |   1  |  0   |   1 |
    rank        | 0  |   1  |  2   |   4 |

    >>You should use rank and not local_rank when using torch.distributed primitives (send/recv etc). local_rank is passed to the training script 
        only to indicate which GPU device the training script is supposed to use.
    >>You should always use rank.
    >>local_rank is supplied to the developer to indicate that a particular instance of the training script should use the “local_rank” GPU device. 
        For illustration, in the example above provided by @spanev, p1 is passed local_rank 0 indicating it should use GPU device id 0. 
    I think this means that p4 passed local_rank 1 to use GPU device id 4

    NOTE: This code is set up to use only 1 Node!!!! Above for future reference if we want to do multi-machine, multi-GPU training with launcher script


'''


fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)


def train(device, dataset, train_loader, model, criterion, optimizer, epochs):

    init_distributed()

    print(
        f"Running backbone DDP on local rank (process): {init_distributed().local_rank}.")
    print(f"This is on network rank (GPU): {init_distributed().rank}.")

    # Convert BatchNorm to SyncBatchNorm.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # create model and move it to GPU with id rank
    model = model.to(init_distributed().local_rank)
    # Now DDP model
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[init_distributed().rank])

    ### START TRAINING LOOP ###
    nr = 0
    current_epoch = 0
    tr_loss = []

    CHECKPOINT_PATH = "/scratch_tmp/$USER/SimCLR_{epoch}.pt"
    if init_distributed().rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()

        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % init_distributed().rank}
        model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))

    for epoch in range(epochs):

        train_loader.sampler.set_epoch(epoch)

        print(f"Epoch [{epoch}/{epochs}]\t")
        stime = time.time()
        model.train()
        tr_loss_epoch = 0

        for step, (x_i, x_j) in enumerate(train_loader):

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                x_i = x_i.to(device)
                x_j = x_j.to(device)

                z_i = model(x_i)
                z_j = model(x_j)

                loss = criterion(z_i, z_j)

            fp16_scaler.scale(loss).loss.backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

            if nr == 0 and step % 5 == 0:
                print(
                    f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

            tr_loss_epoch += loss.item()

        if nr == 0:
            tr_loss.append(tr_loss_epoch / len(dataset))
            print(
                f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(dataset)}\t")
            current_epoch += 1

        time_taken = (time.time()-stime)/60
        print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")


    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if init_distributed().rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def backbone_pretraining(device, DATASET_PATH="/unlabeled/unlabeled/", BATCH_SIZE=16, TEMPERATURE=0.5, NUM_WORKERS=2, SHUFFLE=True, IMAGE_SIZE=112, S=1.0, EPOCHS=1, LR=0.2, MOMENTUM=0.9, WEIGHT_DECAY=1e-6):

    dataset, train_loader = unlabeled_dataloader(
        BATCH_SIZE, NUM_WORKERS, SHUFFLE, DATASET_PATH, IMAGE_SIZE, S)
    model = PreModel('resnet50').to(device)
    criterion = SimCLR_Loss(BATCH_SIZE, TEMPERATURE)
    optimizer = LARS(model.parameters(), lr=LR, momentum=MOMENTUM,
                     weight_decay=WEIGHT_DECAY, max_epoch=EPOCHS)
    # TODO see why SGD fails?
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #train(device, dataset, train_loader, model, criterion, optimizer, EPOCHS)
    mp.spawn(train, nprocs=init_distributed().world_size, args=(init_distributed().world_size,)) # this right? 
