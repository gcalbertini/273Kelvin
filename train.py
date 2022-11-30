import argparse
import warnings
import torch

from fastrcnn import get_model
from labeled_dataloader import labeled_dataloader
from utils import train_one_epoch
from eval import evaluate
from torch.multiprocessing import cpu_count


def train(args):

    model = get_model(args, backbone=args.backbone, num_classes=args.classes) # if you want to train with mobileye backbone, then: get_model(backbone=None)

    _, train_dataloader = labeled_dataloader(args.batch_size, cpu_count()//2, args.shuffle, args.path_lbl, SPLIT="training")
    _, validation_dataloader = labeled_dataloader(args.batch_size, cpu_count()//2, args.shuffle, args.path_lbl, SPLIT="validation")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learn_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # Use a learning rate scheduler: this means that we will decay the learning rate every <step_size> epoch by a factor of <gamma>
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    if not args.verbose_off:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        print('Model Summary:')
        print(model)
        

    print("\nFastRCNN Training START")
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, args.print_freq)
        lr_scheduler.step()
        evaluate(model, validation_dataloader, device)
    torch.save(model.state_dict(), f"./saved_models/model__mom_{args.momentum}_decay_{args.weight_decay}_epochs_{epoch+1}_lr_{args.learn_rate}_backbone_{args.backbone}.pt")

    return model

def main():
    #NOTE The store_true option automatically creates a default value of False. Likewise, store_false will default to True when the command-line argument is not present.

    # Verbose_off action is store_false
    # Example python main.py --verbose_off --> Will not be verbose
    # Example python main.py               --> Will be verbose

    # Train_backbone action is store_true
    # Example python main.py                   --> Will not train backbone
    # Example python main.py --train_backbone  --> Trains backbone


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
    parser.add_argument('-bs', '--batch_size', default=2, type=int, metavar='BATCH_SIZE', help='Batch size to use; default is 32')
    parser.add_argument('-e', '--epochs', default=1, metavar='EPOCHS', type=int, help="Default number of epochs")
    parser.add_argument('-lr', '--learn_rate', default=0.001, metavar='LEARN_RATE', type=float, help="Default learning rate")
    parser.add_argument('-mom', '--momentum', default=0.9, metavar='MOMENTUM', type=float, help="Default momentum")
    parser.add_argument('-dk', '--weight_decay', default=0.0005, metavar='WEIGHT_DECAY', type=float, help="Default weight decay")
    parser.add_argument('-p', '--print_freq', default=200, metavar='PRINT_FREQ', type=int, help="Default print frequency")
    parser.add_argument('-s', '--step', default=50, metavar='SCHEDULER_STEP', type=int, help="Default step size for scheduler")
    parser.add_argument('-g', '--gamma', default=0.2, metavar='SCHEDULER_GAMMA', type=float, help="Default gamma factor for scheduler")
    parser.add_argument('-f', '--freeze', action='store_true', help='Freeze backbone weights; default is False')

    #=====================SIMCLR ONLY: EDIT THESE FOR BACKBONE TRAIN RUN===============================================================================
    # NOTE these will come into play after --train_backbone is specified and doing something like: python train.py --train_backbone -bbe 5 -bbs 12345 --backbone_lr 1e-4

    parser.add_argument('-bbe','--backbone_epochs', default=1, metavar='BACKBONE_EPOCHS', type=int, help="Default number of backbone epochs")
    parser.add_argument('-bbc','--backbone_cuda', action='store_false', help="Use CUDA for backbone training")
    parser.add_argument('-bbsd','--backbone_seed', default=77777, metavar='BACKBONE_SEED', type=int, help="Backbone seed for reproducibility")
    parser.add_argument('-bbimg','--backbone_img_size', default=224, metavar='BACKBONE_IMG_SIZE', type=int, help="Backbone img size")
    parser.add_argument('-bbsv','--backbone_save_directory', default='saved_models/', metavar='BACKBONE_SAVE_DIR_PATH', type=str, help="Backbone save checkpoint directory path")
    parser.add_argument('-bblp','--backbone_load_pretrained', action='store_true', help="Backbone load pretraining")
    parser.add_argument('-bbg','--backbone_grad_accumulate_steps', type=int, default = 5, metavar='BACKBONE_GRAD_ACCUM_STEPS', help="Backbone gradient accumulation steps")
    parser.add_argument('-bbbs', '--backbone_batch_size', default=200, type=int, metavar='BACKBONE_BATCH_SIZE', help='Backbone batch size to use; default is 200')
    parser.add_argument('-bbemb','--backbone_embedding_size', type=int, default = 128, metavar='BACKBONE_EMBED_SIZE', help='Backbone embedding size')
    parser.add_argument('-bblr','--backbone_lr', type=float, default = 3e-4, metavar='BACKBONE_ADAM_LEARN_RATE', help='Backbone learning rate for ADAM')
    parser.add_argument('-bbdk','--backbone_weight_decay', type=float, default = 1e-6, metavar='BACKBONE_WEIGHT_DECAY', help='Backbone weight decay')
    parser.add_argument('-bbtmp','--backbone_temperature', type=float, default=0.1, metavar='BACKBONE_TEMP', help='Backbone temperature parameter (set to 0.1 or 0.5)')
    parser.add_argument('-bbcp','--backbone_checkpoint_path', default='./SimCLR_ResNet18.ckpt', metavar='BACKBONE_CHECKPOINT_PATH', type=str, help="Backbone checkpoint path")
    parser.add_argument('-bbr','--backbone_resume', action='store_true', help="Backbone resume training from checkpoint; default is False")

    args = parser.parse_args()
    print()
    print(args)

    train(args)

if __name__=="__main__":
    main()
