from torch.autograd import Function
import torch.nn as nn
import torch



class LatentLoss(nn.Module):    
    def __init__(self):
        super(LatentLoss, self).__init__()

    def forward(self, mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        latentLoss = torch.sum(KLD_element).mul_(-0.5)
        return latentLoss
