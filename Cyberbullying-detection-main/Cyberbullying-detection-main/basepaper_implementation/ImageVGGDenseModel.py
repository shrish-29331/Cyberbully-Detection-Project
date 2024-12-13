import torch
import torch.nn as nn

class ImageVGGDenseModel(nn.Module):
    def __init__(self, input_dim):
        super(ImageVGGDenseModel, self).__init__()
        self.Linear_input = nn.Linear(input_dim, 25088) # input_dim: 25088
        self.Linear_25088_1024 = nn.Linear(25088, 1024)
        self.Linear_1024_512 = nn.Linear(1024, 512)
        self.Linear_512_256 = nn.Linear(512, 256)
        self.Linear_256_128 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.leakyReLU = nn.LeakyReLU(0.01)
        self.ReLU = nn.ReLU()
        
        
    def forward(self, x):
        x = self.Linear_input(x)
        x = self.leakyReLU(x)
        x = self.Linear_25088_1024(x)
        x = self.leakyReLU(x)
        x = self.Linear_1024_512(x)
        return x