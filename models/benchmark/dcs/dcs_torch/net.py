import torch.nn as nn
import torch
import torch.nn.functional as F

class MLPGenerator(nn.Module):
    def __init__(self,input_size=16):
        super(MLPGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 32),
            nn.Tanh(),
        )
        #self.scaler = nn.Parameter(torch.randn(1))

    def forward(self,x):
        x = F.normalize(x, p=2, dim=-1)
        x = self.net(x)
        #x = self.scaler * x
        return x