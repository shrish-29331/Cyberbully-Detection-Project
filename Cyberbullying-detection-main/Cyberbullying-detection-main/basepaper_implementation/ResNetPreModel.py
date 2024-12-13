import torch
import torch.nn as nn
from torchvision import transforms, models

class ResNetPreModel(nn.Module):
    def __init__(self):
        super(ResNetPreModel, self).__init__()
        # Load pre-trained ResNet model for image processing
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 512)  # Modify the final layer
        
    def forward(self, image):
        x = self.image_model(image)
        return x