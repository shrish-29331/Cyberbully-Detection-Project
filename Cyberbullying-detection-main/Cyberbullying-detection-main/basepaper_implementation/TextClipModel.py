import torch
import torch.nn as nn

class TextClipModel(nn.Module):
    def __init__(self, input_dim):
        super(TextClipModel, self).__init__()
        self.Linear_input = nn.Linear(input_dim, 512)
        self.Linear_512_256 = nn.Linear(512, 256)
        self.Linear_256_128 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.leakyReLU = nn.LeakyReLU(0.01)
        self.ReLU = nn.ReLU()
        
        
    def forward(self, x):
        x = self.Linear_input(x)
        x = self.leakyReLU(x)
        x = self.Linear_512_256(x)
        x = self.leakyReLU(x)
        x = self.Linear_256_128(x)
        return x