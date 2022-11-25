import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
from torch import optim

from unlabeled_dataloaders import data_loader
from unlabeled_dataloaders import show

# simply returns input
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# defines one linear layer only, and whether its followed by batch normalisation
class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn # batch normalisation
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn) # use bias only when batch norm is false, why?
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

# this is the projection head g(.), which sits on top of ResNet
# https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/4
# above useful if getting this error: ValueError: Expected more than 1 value per channel when training, got input size [1, X]

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x

class PreModel(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        
        #PRETRAINED MODEL - don't make it pretrained
        self.pretrained = models.resnet50(weights=None)
        
        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = Identity()
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
        
        self.projector = ProjectionHead(2048, 2048, 128)

    def forward(self,x):
        out = self.pretrained(x)
        xp = self.projector(torch.squeeze(out))
        
        return xp

#https://github.com/Spijkervet/SimCLR/blob/cd85c4366d2e6ac1b0a16798b76ac0a2c8a94e58/simclr/modules/nt_xent.py#L7

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, z_i, z_j):

          N = 2 * self.batch_size

          z = torch.cat((z_i, z_j), dim=0)

          sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

          sim_i_j = torch.diag(sim, self.batch_size)
          sim_j_i = torch.diag(sim, -self.batch_size)
          
          # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
          positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
          negative_samples = sim[self.mask].reshape(N, -1)
          
          #SIMCLR
          labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
          
          logits = torch.cat((positive_samples, negative_samples), dim=1)
          loss = self.criterion(logits, labels)
          loss /= N
          
          return loss

if __name__ == "__main__":

    BATCH_SIZE = 16
    NUM_WORKERS = 2
    SHUFFLE = True

    LEN_DATSET=1000 # to delete later
    PATH = "./unlabeled_test/" # to delete later
    train_loader = data_loader(BATCH_SIZE, NUM_WORKERS, SHUFFLE, PATH)

    model = PreModel('resnet50').to('cuda:0')
    criterion = SimCLR_Loss(BATCH_SIZE, temperature=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # N x C x H x W
    #tensor = torch.rand(2, 3, 224, 224) # make sure batch size is never 1!
    #tensor = tensor.to('cuda:0')
    #out = model(tensor)

    ### TRAINING ###
    import time

    nr = 0
    current_epoch = 0
    epochs = 10
    tr_loss = []
    val_loss = []

    for epoch in range(epochs):
            
        print(f"Epoch [{epoch}/{epochs}]\t")
        stime = time.time()

        model.train()
        tr_loss_epoch = 0
        
        for step, (x_i, x_j) in enumerate(train_loader):

            optimizer.zero_grad()

            x_i = x_i.to('cuda:0')
            x_j = x_j.to('cuda:0')

            #show((x_i[0] * 255).to(torch.uint8)) 
            #show((x_i[1] * 255).to(torch.uint8))
            #show((x_j[0] * 255).to(torch.uint8))
            #show((x_j[1] * 255).to(torch.uint8))

            # positive pair, with encoding
            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()
            
            if nr == 0 and step % 5 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

            tr_loss_epoch += loss.item()

        if nr == 0:
            tr_loss.append(tr_loss_epoch / LEN_DATSET)
            print(f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / LEN_DATSET}\t")
            current_epoch += 1
        
        time_taken = (time.time()-stime)/60
        print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")



