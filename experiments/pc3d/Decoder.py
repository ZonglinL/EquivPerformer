import torch
from torch import nn
from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias, GConvSE3, GMaxPooling, GAvgPooling,GNormSE3, AttentionPooling
from equivariant_attention.fibers import Fiber


class Decoder(nn.Module):

    def __init__(self,dim,num_class):
        super().__init__()
        self.dim = dim
        self.num_class = num_class
        self.proj = nn.Linear(self.dim,1)
        self.decoder = nn.Sequential(nn.Linear(self.dim,self.dim),nn.ReLU(),nn.Linear(self.dim,self.num_class))
    @profile
    def forward(self, x):
        """

        :param x: Batch N dim
        :return:
        """
        attention = self.proj(x).softmax(-2) ## Batch N 1
        x = torch.matmul(attention.transpose(-1,-2),x).view(-1,self.dim) # Batch dim
        x = self.decoder(x) ## batch class

        return x

