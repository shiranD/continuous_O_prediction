import torch.nn as nn
import torch
from space import build_space
from scipy.special import ive
from scipy import signal
import pdb
import torch.distributions.normal as normal

class SIMLoss_sqrt(torch.nn.Module): # for gpu performance

    def __init__(self, dim=0):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, input, target):
        return torch.sqrt(2*(1-self.cos(input, target))).mean()

