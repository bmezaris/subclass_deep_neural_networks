"""
Implementation of the VGG16 network architecture in PyTorch as described in the paper:
Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition.
In: ICLR. San Diego, CA, USA (May 2015)

Code obtained from:
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.penultim_layer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.penultim_layer(output)
        output = self.classifier(output)
    
        return output

def make_layers(cfg, batch_norm=False, input_channel=3):
    layers = []

    #input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def vgg16_bn(inputChannel=3, numClass=100):
    return VGG(make_layers(cfg['D'], batch_norm=True, input_channel=inputChannel), num_class=numClass)
