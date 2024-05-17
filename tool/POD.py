import torch
from torch.nn import Parameter
import torch.nn as nn
import pickle
import torch.nn.functional as F
import numpy as np
import os

# 计算余弦距离
class CosineLinear_PEDCC(nn.Module):
    def __init__(self, in_features, out_features, config):
        super(CosineLinear_PEDCC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)  # Parameter将张量变为可训练的参数 ，tensor为in_features*out_features
        #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        map_dict = read_pkl(config.pedcc_path)
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.out_features):
            # tensor_map_dict = torch.from_numpy(map_dict[label_index])
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
        label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = label_40D_tensor
        #print(self.weight.data)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
        cos_theta = x.mm(w)  # size=(B,Classnum)  x.dot(ww) #x与w相乘

        return cos_theta  # size=(B,Classnum)


def read_pkl(PEDCC_PATH):
    #pedcc_path = os.path.join(conf.HOME, PEDCC_PATH)
    f = open(PEDCC_PATH, 'rb')
    a = pickle.load(f)
    f.close()
    return a


#####NaCLoss##################
def NaCLoss(input, target, delta):
    ret_before = torch.mul(input, target)
    ret_before = torch.sum(ret_before, dim=1).view(-1, 1)  # 行求和

    add_feature = delta * torch.ones((input.shape[0], 1)).cuda()
    input_after = torch.cat((input, add_feature), dim=1)  # 行拼接
    input_after_norm = torch.norm(input_after, p=2, dim=1, keepdim=True)

    ret = ret_before / input_after_norm
    ret = 1 - ret
    ret = ret.pow(2)
    ret = torch.mean(ret)

    return ret


def SCLoss(feature, average_feature):
    feature_norm = l2_norm(feature)
    feature_norm = feature_norm - average_feature
    covariance100 = 1 / (feature_norm.shape[1] - 1) * torch.mm(feature_norm.T, feature_norm).float()
    covariance100_loss = torch.sum(pow(covariance100, 2)) - torch.sum(pow(torch.diagonal(covariance100), 2))
    covariance100_loss = covariance100_loss / (covariance100.shape[0] - 1)
    return covariance100_loss

def l2_norm(input, dim=1):
    norm = torch.norm(input, 2, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output
