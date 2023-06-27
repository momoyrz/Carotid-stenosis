import math
import numpy as np
import scipy.sparse as sp

import torch

from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import models

from models import enet
from utility.preprocessing import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, singal=None, norm='', bias=True):
        super(GraphConvolution, self).__init__()
        if singal == 'self':
            self.linear = nn.Linear(in_features, out_features, bias)
        else:
            self.linear1 = nn.Linear(in_features, 640, bias)
            self.linear2 = nn.Linear(640, out_features, bias)
            self.norm = norm
            self.a = nn.ReLU()
        
    def forward(self, x, adj=1.0):
        x = to_dense(x)
        if isinstance(adj, (float, int)):
            x2 = self.linear(x)
        else:
            adj = adj_norm(adj, True) if self.norm == 'symmetric' else adj_norm(adj,
                                                                                False) if self.norm == 'asymmetric' else adj

            x1 = self.a(self.linear1(torch.matmul(adj, x)))
            x2 = self.linear2(x1)
        return x2

    
class ImageGraphConvolution(nn.Module):
    """
    GCN layer for image data
    """

    def __init__(self, enc , out_dim=14, inchannel=3):
        super(ImageGraphConvolution, self).__init__()
        self.encoder = enc
        self.classifier = nn.Linear(1024, out_dim)

    def forward(self, input, adj=1.0):
        x = self.encoder(input).squeeze()
        x = x.view(-1, 1024)
        support = self.classifier(x) 
        if isinstance(adj, (float, int)):
            output = support*adj  
        else:
            output = torch.spmm(adj, support) 
        return output

class MyAlexNet(nn.Module):
    def __init__(self,outnum=14, gpsize=4, inchannel=3):
        super(MyAlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        if inchannel!=3:
            original_model.features[0]=nn.Conv2d(1,64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.features = original_model.features        
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(256, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

class MyResNet34(nn.Module):
    def __init__(self):
        super(MyResNet34, self).__init__()
        original_model = models.resnet34()
        original_model.load_state_dict(torch.load('/home/ubuntu/qujunlong/pre_weights/resnet34.pth'))
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        return x
    
class MyResNet50(nn.Module):
    def __init__(self,outnum=14,gpsize=4, inchannel=3):
        super(MyResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True) 
        if inchannel != 3:
            original_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(2048, 1024, 3, padding=1),
                                                          nn.BatchNorm2d(1024),
                                                          nn.ReLU(inplace=True),
                                                          nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


class MyResNet50_1(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyResNet50_1, self).__init__()
        original_model = models.resnet50(pretrained=True)
        if inchannel != 3:
            original_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(2048, 256, 3, padding=1),
                                                          nn.BatchNorm2d(256),
                                                          nn.ReLU(inplace=True),
                                                          nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

class MyEfficientB0(nn.Module):
    def __init__(self):
        super(MyEfficientB0, self).__init__()
        original_model = enet.efficientnet_b0()
        original_model.load_state_dict(torch.load('/home/ubuntu/qujunlong/pre_weights/efficientnetb0.pth'))
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1280)
        return x

class MyDensNet161(nn.Module):
    def __init__(self, outnum=14,gpsize=4, inchannel=3):
        super(MyDensNet161, self).__init__()
        original_model = models.densenet161(pretrained=True)
        if inchannel!=3:
            original_model.features.conv0=nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(2208, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet201(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyDensNet201, self).__init__()
        original_model = models.densenet201(pretrained=True)
        if inchannel!=3:
            original_model.features.conv0=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(1920, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet121(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyDensNet121, self).__init__()
        original_model = models.densenet121(pretrained=True)
        if inchannel!=3:
            original_model.features.conv0=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x    

