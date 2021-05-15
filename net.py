import torch
import torch.nn as nn
from torchvision import models

'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16 = vgg16.features
        for param in vgg16.parameters():
            param.requires_grad_(False)
        self.vgg16=vgg16


        self.classifier=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512*7*7,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256,30),
            nn.Softmax(dim=1)
        )


    def forward(self,x):
        x=self.vgg16(x)
        x=x.view(x.size(0),-1)
        output=self.classifier(x)
        return output
'''

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        for param in resnet50.parameters():
            param.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 30),
        )

    def forward(self,x):
        x = self.features(x)
        x= torch.flatten(x,1)
        x = self.classifier(x)

        return x



