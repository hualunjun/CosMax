import torch
import torchvision
import os
from torch import nn
import numpy as np
import copy

def load_swav(config):
    if os.path.exists(config.pretrain_model_path):
        model = torchvision.models.resnet50(pretrained=True)
        # model = torch.load(config.pretrain_model_path)
    else:
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        torch.save(model, config.pretrain_model_path)

    model.fc = nn.Identity()
    return model

def load_barlowtwins(config):
    if os.path.exists(config.pretrain_model_path):
        model = torch.load(config.pretrain_model_path)
    else:
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        torch.save(model, config.pretrain_model_path)

    model.fc = nn.Identity()
    # model.fc = nn.BatchNorm1d(2048, affine=False)

    return model

class incrementalDiscriminatorHead(torch.nn.Module):
    def __init__(self, config, preWeight=None):
        super(incrementalDiscriminatorHead, self).__init__()
        bias = torch.from_numpy(np.load("./data/imagenetCenter.npy")).cuda()
        bias = bias.float()
        self.bias = nn.Parameter(bias, requires_grad=False)
        fc = nn.Linear(2048, config.classNum - config.preClassNum, bias=False)
        w = fc.weight.data.t()
        self.currentWeight = nn.Parameter(w, requires_grad=True)

        if preWeight is None:
            self.preWeight = None
        else:
            self.preWeight = nn.Parameter(preWeight, requires_grad=False)

    def forward(self, x):
        out = x - self.bias
        outNorm = torch.linalg.norm(out, dim=1, keepdim=True)
        outNorm = outNorm.clamp(min=1e-12)
        out = torch.div(out, outNorm)

        weightNorm = torch.linalg.norm(self.currentWeight, dim=0, keepdim=True)
        weightNorm = weightNorm.clamp(min=1e-12)
        weight = torch.div(self.currentWeight, weightNorm)
        currentOut = torch.mm(out, weight)
        if self.preWeight is None:
            out = currentOut
        else:
            weightNorm = torch.linalg.norm(self.preWeight, dim=0, keepdim=True)
            weightNorm = weightNorm.clamp(min=1e-12)
            weight = torch.div(self.preWeight, weightNorm)
            preOut = torch.mm(out, weight)

            out = torch.cat((preOut, currentOut), 1)

        return out

def convert_relu_to_activition(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, torch.nn.LeakyReLU(0.1))
        else:
            convert_relu_to_activition(child)

class MisakaNet(torch.nn.Module):
    def __init__(self, config, preWeight=None):
        super(MisakaNet, self).__init__()
        # self.backbone = load_barlowtwins(config)
        self.backbone = load_barlowtwins(config)
        self.fc = incrementalDiscriminatorHead(config, preWeight)


    def forward(self, x):
        xout = self.backbone(x)
        out = self.fc(xout)

        return out, xout


def model_load(model_path):
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print('load success! ')
    else:
        print('No model is saved in \'{}\''.format(model_path))
        model = None
    return model


def model_save(model, model_path, isDataParall):
    if isDataParall:
        model = model.module

    torch.save(model, model_path)
    print('model is saved')

class myFC(torch.nn.Module):
    def __init__(self, b, w):
        super(myFC, self).__init__()
        bias = b
        bias = bias.float()
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.weight = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        out = x - self.bias
        outNorm = torch.linalg.norm(out, dim=1, keepdim=True)
        outNorm = outNorm.clamp(min=1e-12)
        out = torch.div(out, outNorm)

        weightNorm = torch.linalg.norm(self.weight, dim=0, keepdim=True)
        weightNorm = weightNorm.clamp(min=1e-12)
        weight = torch.div(self.weight, weightNorm)
        out = torch.mm(out, weight)

        return out

def saveFinalModel(model, MisakaNum):
    final = torch.load("./../../barlowtwins.pth")
    final = copy.deepcopy(model.barlowtwins)
    if model.fc.preWeight is None:
        weight = model.fc.currentWeight.data
    else:
        weight = torch.cat((model.fc.preWeight.data, model.fc.currentWeight.data), 1)
    final.fc = myFC(model.fc.bias.data, weight)
    torch.save(final, "./model/" + MisakaNum +"-final.pth")

class MisakaNet_Teacher(nn.Module):
    def __init__(self, model):
        super(MisakaNet_Teacher, self).__init__()

        self.backbone = copy.deepcopy(model)
        self.backbone.fc = nn.Identity()
        self.fc = copy.deepcopy(model.fc)

    def forward(self, x):
        xout = self.backbone(x)
        out = self.fc(xout)

        return out, xout
