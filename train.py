import argparse
import sys
import warnings
import torch
from torch.multiprocessing import cpu_count
from tqdm import tqdm
import math
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastrcnn import get_model
from labeled_dataloader import labeled_dataloader
#from utils import train_one_epoch
from eval import evaluate
import warnings

warnings.filterwarnings("ignore")

def train(backbone="SimCLR", BATCH_SIZE=2, EPOCHS=50, NUM_WORKERS=cpu_count()//2, SHUFFLE=True, DATASET_PATH="/labeled/labeled", LR=0.008, MOM=0.9, DECAY=1e-4):

def train(args):
    #This function handles multi-gpu training 
    model = get_model(args, backbone=args.backbone, num_classes=args.classes) # if you want to train with mobileye backbone, then: get_model(backbone=None)

    _, train_dataloader = labeled_dataloader(BATCH_SIZE, NUM_WORKERS, SHUFFLE, DATASET_PATH, SPLIT="training")
    _, validation_dataloader = labeled_dataloader(1, NUM_WORKERS, False, DATASET_PATH, SPLIT="validation") # BATCH=1

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    #print(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=DECAY, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-3, amsgrad=True)

    # Use a learning rate scheduler: this means that we will decay the learning rate every <step_size> epoch by a factor of <gamma>
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    print("!!! FastRCNN Training START !!!")
    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch)
        if (epoch) % 2 == 0:
            torch.save(model.state_dict(), f"./save_fastrcnn_models/model_3_batch_{BATCH_SIZE}_mom_{MOM}_decay_{DECAY}_epochs_{epoch}_lr_{LR}_backbone_{backbone}_RPN.pt")
        if (epoch) % 5 == 0:
            evaluate(model, validation_dataloader, device)

    evaluate(model, validation_dataloader, device)
    torch.save(model.state_dict(), f"./save_fastrcnn_models/model_3_batch_{BATCH_SIZE}_mom_{MOM}_decay_{DECAY}_epochs_{epoch}_lr_{LR}_backbone_{backbone}_RPN.pt")

    return model

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []

    #i = 0
    
    for images, targets in tqdm(loader):
        #print("loop")
        images = list(image.to(device) for image in images)
        #targets = [{k: v for k, v in t.items()} for t in targets]
        #targets = [{k: v.clone().detach().to(device) for k, v in t.items()} for t in targets]
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        #print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        #if i % 750 == 0:
        #   print(loss_value)
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(targets)
            print(loss_dict)
            print(f"Loss is {loss_value}, stopping training") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        #i+=1
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

def main():
    #NOTE The store_true option automatically creates a default value of False. Likewise, store_false will default to True when the command-line argument is not present.

    # Verbose_off action is store_false
    # Example python main.py --verbose_off --> Will not be verbose
    # Example python main.py               --> Will be verbose

    # Train_backbone action is store_true
    # Example python main.py                   --> Will not train backbone
    # Example python main.py --train_backbone  --> Trains backbone

    #NOTE BATCH SIZES SHOULD BE A MULTIPLE OF GPUs USED AND GREATER THAN THE NUMBER OF GPUs. THE EFFECTIVE BATCH SIZE/GPU IS BATCH_SIZE_SPECIFIED*GRAD_ACCUM_STEPS,
    # WHERE TOTAL EFFECTIVE BATCH SIZE YOU RUN IS THIS NUMBER MULTIPLIED NUMBER OF GPUs FOUND

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_lbl', default="/labeled/labeled", metavar='DATA_PATH_LBL', type=str, help="Default path for labeled data; default used is '/labeled/labeled'; note toy set's is 'labeled_data/'")
    parser.add_argument('--path_unlbl', default="/unlabeled/unlabeled", metavar='DATA_PATH_UNLBL', type=str, help="Default path for unlabeled data; default used is'/unlabeled/unlabeled'; note toy set's is 'unlabeled_data/'")
    parser.add_argument('--shuffle', action='store_true', help="Shuffle data toggle")
    parser.add_argument('-voff', '--verbose_off', action='store_false', help="Verbose mode toggle")
    parser.add_argument('-c', '--classes', default=100, type=int, metavar='NUM_CLASSES', help='Number of classes; default is 100')

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
    parser.add_argument('-f', '--freeze', action='store_true', help='Freeze backbone weights; default is False')

    #=====================SIMCLR ONLY: EDIT THESE FOR BACKBONE TRAIN RUN===============================================================================
    # NOTE these will come into play after --train_backbone is specified and doing something like: python train.py --train_backbone -bbe 5 -bbbs 12345 --backbone_lr 1e-4

    parser.add_argument('-bbe','--backbone_epochs', default=10, metavar='BACKBONE_EPOCHS', type=int, help="Default number of backbone epochs")
    parser.add_argument('-bbcoff','--backbone_cuda_off', action='store_false', help="Toggle CUDA for backbone training")
    parser.add_argument('-bbsd','--backbone_seed', default=77777, metavar='BACKBONE_SEED', type=int, help="Backbone seed for reproducibility")
    parser.add_argument('-bbimg','--backbone_img_size', default=224, metavar='BACKBONE_IMG_SIZE', type=int, help="Backbone img size")
    parser.add_argument('-bbsv','--backbone_save_directory', default='saved_models/', metavar='BACKBONE_SAVE_DIR_PATH', type=str, help="Backbone save checkpoint directory path")
    parser.add_argument('-bblp','--backbone_load_pretrained', action='store_true', help="Backbone load pretraining")
    parser.add_argument('-bbg','--backbone_grad_accumulate_steps', type=int, default = 1, metavar='BACKBONE_GRAD_ACCUM_STEPS', help="Backbone gradient accumulation steps")
    parser.add_argument('-bbbs', '--backbone_batch_size', default=4, type=int, metavar='BACKBONE_BATCH_SIZE', help='Backbone batch size to use')
    parser.add_argument('-bbemb','--backbone_embedding_size', type=int, default = 128, metavar='BACKBONE_EMBED_SIZE', help='Backbone embedding size')
    parser.add_argument('-bblr','--backbone_lr', type=float, default = 3e-4, metavar='BACKBONE_ADAM_LEARN_RATE', help='Backbone learning rate for ADAM')
    parser.add_argument('-bbdk','--backbone_weight_decay', type=float, default = 1e-6, metavar='BACKBONE_WEIGHT_DECAY', help='Backbone weight decay')
    parser.add_argument('-bbtmp','--backbone_temperature', type=float, default=0.1, metavar='BACKBONE_TEMP', help='Backbone temperature parameter (set to 0.1 or 0.5)')
    parser.add_argument('-bbcp','--backbone_checkpoint_path', default='./SimCLR_ResNet18.ckpt', metavar='BACKBONE_CHECKPOINT_PATH', type=str, help="Backbone checkpoint path")
    parser.add_argument('-bbr','--backbone_resume', action='store_true', help="Backbone resume training from checkpoint; default is False")

    args = parser.parse_args()
    DIR = os.path.join(args.path_lbl, "training", "images")
    image_dir_training_len =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    print(f'Found {image_dir_training_len} training images.')
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    available_gpus = len(available_gpus)

    # NOTE this is only for backbone dim checking
    try: 
        effective_batch_size = args.backbone_batch_size*args.backbone_grad_accumulate_steps
        assert image_dir_training_len % effective_batch_size == 0

        print('Dimensions for backbone OK. Note that code for training backbone computes effective batch size per GPU.')
        params = vars(args)
        print(params)
        print(f'Total effective batch size (across all GPU(s)) is {effective_batch_size*available_gpus}')
        train(args)

    except: 
        print(f'Epoch steps must be integer but is {image_dir_training_len/effective_batch_size}; i.e. len(data) / effective batch size needs to be a whole number')
        print('Attempting to change grad_accum_steps of backbone to 1 to resolve error...')
        args.backbone_grad_accumulate_steps = 1
        new_effective_batch_size = args.backbone_batch_size*args.backbone_grad_accumulate_steps
        print(f'Epoch steps now is {image_dir_training_len/new_effective_batch_size}')
        assert image_dir_training_len % new_effective_batch_size == 0

        print('Adjusted dimensions for backbone OK. Note that code for training backbone computes effective batch size per GPU.')
        params = vars(args)
        print(params)
        print(f'Total effective batch size (across all GPU(s)) is {new_effective_batch_size*available_gpus}')
        train(args)

    finally:
        print('Could not adjust parameters.')
        sys.exit()

    

if __name__=="__main__":
    main()