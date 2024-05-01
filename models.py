import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import torchvision.models as torchmodels


class CustomClip(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.clip_model = clip_model
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        with torch.no_grad():
            img_fts = self.clip_model.encode_image(x)
        img_fts  = img_fts.type(torch.float)
        out1 = F.relu(self.fc1(img_fts))
        out2 = self.fc2(out1)
        return out2

# ResNet Classifier
class Resnet_Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Use resnet18 + one fc layer
        self.resnet = torchmodels.resnet18(pretrained=True)
        in_ftrs = self.resnet.fc.in_features
        self.fc1 = nn.Linear(in_ftrs,1)

    def forward(self, clip_model, x):
        x = self.relu(self.resnet(x))
        x = self.fc1(x)

        return x

# 