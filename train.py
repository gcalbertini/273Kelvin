import numpy as np
import shutil, time, os, requests, random, copy
import gc
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from network import PreModel

model = PreModel('resnet50').to('cuda:0')