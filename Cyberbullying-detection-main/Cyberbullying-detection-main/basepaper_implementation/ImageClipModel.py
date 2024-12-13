import torch
import torch.nn as nn

class ImageClipModel(nn.Module):
    def __init__(self,input_dim):
        super(ImageClipModel, self).__init__()
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
    
# def test_model():
#     image_clip_model = ImageClipModel(512).to(device)
#     data_sample = dataset_clip_vgg[0]
#     image_clip_data = data_sample["image_clip_input"]
#     print(image_clip_data.shape)
#     y = image_clip_model(image_clip_data)
#     y.shape
        