
import numpy as np
# import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = torch.ones((batch_size, batch_size), dtype=bool).fill_diagonal_(0)
        self.criterion = nn.CrossEntropyLoss()
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.N = 2*batch_size

    def forward(self, z_i, z_j):

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        ''' Note: **
        Cosine similarity matrix of all samples in batch:
        a = z_i
        b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|
        Postives:
        Diagonals of ab and ba '\'
        Negatives:
        All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.N, 1)
        negative_samples = sim[self.mask].reshape(self.N, -1)
        
        #SIMCLR
        labels = torch.zeros(self.N).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        
        return loss

    # def mask_samples(self):
    #     mask = torch.ones((self.N, self.N), dtype=bool)
    #     mask = mask.fill_diagonal_(0)
        
    #     for i in range(self.batch_size):
    #         mask[i, self.batch_size + i] = 0
    #         mask[self.batch_size + i, i] = 0
    #     return mask