from sys import path

import numpy
import torch
import numpy as np
# path.append('./../data')
# path.append('./../model')
# path.append('./../tool')

from config import *
from data.DataSet import *
from model.modelseting import *
import pandas as pd
from torch import nn, optim





def interface_main():

    cfg = Config()
    misaka = MisakaNet(cfg)

    misaka = torch.nn.DataParallel(misaka).cuda()  # 多卡

    optimizer = optim.AdamW(misaka.parameters(), lr=cfg.lr)
    start_epoch, valid_best_acc = model_state_load(cfg.model_path, misaka, optimizer)
    print("epoch:", start_epoch)
    print("valid_best_acc:", valid_best_acc)

    centers = misaka.module.fc.weight
    npc = centers.cpu().detach().numpy()
    np.savetxt('cosMax+0-001-caltech256.csv', npc, delimiter=',', fmt='%f')
    # for i in range(2, 3):
    #     cfg = Config()
    #
    #     misaka = MisakaNet(cfg)
    #     misaka = torch.nn.DataParallel(misaka).cuda()
    #     load_trained_misaka(misaka, cfg)
    #     print(misaka)
    #     Dataset = ImageFolder(cfg.trainDir, transform=cfg.train_transformer)  # data_dir精确到分类目录的上一级
    #     dataloader = DataLoader(Dataset, batch_size=cfg.batchsz, shuffle=False)
    #     interface_and_get_feature(misaka, cfg, dataloader, dataType="train", fi=i)



def interface_and_get_feature(model, config, input_data, dataType, fi):
    feature_data = np.array([])
    label_data = np.array([])
    norm_data = np.array([])

    model.cuda()
    model.eval()
    with torch.no_grad():
        # interface
        for x, label in input_data:
            x, label = x.cuda(), label.cuda()

            xout, xnorm = model(x)


            label_array = label.cpu().numpy()
            # label_array = numpy.ones(label.shape)*config.main_c
            if label_data.size == 0:
                label_data = label_array
            else:
                label_data = np.concatenate((label_data, label_array), axis=0)

            norm_array = xnorm.squeeze().cpu().numpy()
            if norm_data.size == 0:
                norm_data = norm_array
            else:
                norm_data = np.concatenate((norm_data, norm_array), axis=0)

            features_array = xout.cpu().numpy()
            if feature_data.size == 0:
                feature_data = features_array
            else:
                feature_data = np.concatenate((feature_data, features_array), axis=0)

            print("label_data")
            print(label_data)

        print("feature_data.shape = ", feature_data.shape)
        print("label_data.shape = ", label_data.shape)
        fl_data = np.insert(feature_data, feature_data.shape[1], label_data, axis=1)
        # fl_data = np.insert(fl_data, fl_data.shape[1], norm_data, axis=1)

        print("fl_data.shape", fl_data.shape)
        np.savetxt('./../Dataset/features/cifar100Cosine/' + dataType + '_by_' + 'Misaka' + config.MisakaNum +'(' + str(fi) + ')' +'.csv',
                   fl_data, delimiter=',', fmt='%f')



def hook(module, inputdata, output):
    global feature_data
    # print(output)
    # print(inputdata[0])
    output_array = inputdata[0].cpu().numpy()
    if feature_data.size == 0:
        feature_data = output_array
    else:
        feature_data = np.concatenate((feature_data, output_array), axis=0)
    # print("feature_data:", feature_data)
    print('feature_data.shape:', feature_data.shape)
    # print(output_array)
    # print(inputdata[0])
    # print(inputdata[0].shape)

def OneInterface(model, ImageDir, ImageName, config, Cneters, base):
    global feature_data
    model.to(config.device)
    model.eval()

    img = Image.open(ImageDir+ImageName)
    transformer = config.test_transformer
    img_t = transformer(img)
    x = torch.unsqueeze(img_t, 0)  # 给最高位添加一个维度，也就是batchsize的大小
    pred = -1
    with torch.no_grad():
        handle = model.resnet.fc.register_forward_hook(hookForSingleImage)
        # interface
        x = x.to(config.device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        handle.remove()
        # 返回值最大的维度值 dim的不同值表示不同维度。特别的在dim=0表示二维中的列，dim=1在二维矩阵中表示行。

    MinDistance = CalculDistance(Center=Cneters[0], x=feature_data)
    MinIndx = 0
    for i in range(1,len(Cneters), 1):
        Distance = CalculDistance(Center=Cneters[i], x=feature_data)
        if Distance<=MinDistance:
            MinDistance = Distance
            MinIndx = i

    CNum = base+MinIndx
    CNumStr = str(CNum)  # 数字转化为字符串
    CNumStrSameLen = CNumStr.zfill(3)
    save_path = "./data/features/cifar10_train_by_Misaka10051/train/"+CNumStrSameLen+"/"
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    img.save(save_path+ImageName)



def hookForSingleImage(module, inputdata, output):
    global feature_data
    output_array = inputdata[0].cpu().numpy()
    feature_data = output_array

def CalculDistance(Center, x):
    d = 0
    # print("Center.shape:", Center.shape)
    # print("Center:", Center)
    # print("x.shape:", x.shape)
    # print("x:", x)
    for i in range(0, len(Center), 1):
        d = d+(Center[i]-x[0, i])**2

    return d

def LoadImageAndResave():
    cfg = Config()
    misaka = MisakaNet(cfg)
    AllCenters = pd.read_csv('./data/features/cifar10_train_by_Misaka10051/Centers.csv', header=None)
    AllCentersArry = np.array(AllCenters)
    FeaturesNum = 2048
    base = 0
    for i in range(0,cfg.class_number,1):
        ImageDir = "./data/Cifar10Image/train/"+str(i)+"/"
        ImageNames = os.listdir(ImageDir)
        ImageNames.sort()
        print("len(ImageNames)", len(ImageNames))
        print("ImageNames", ImageNames)
        indx = (AllCentersArry[:, FeaturesNum] == i)
        # print(indx)
        Cneters = AllCentersArry[indx, 0:FeaturesNum]
        for j in range(0,len(ImageNames), 1):
            OneInterface(model=misaka, ImageDir=ImageDir, ImageName=ImageNames[j], config=cfg, Cneters=Cneters, base=base)

        base = base+len(Cneters)
        print(base)


def load_trained_misaka(model, config):
    if os.path.exists(config.model_path):
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model'])
        print('load trained {} success! '.format(config.MisakaNum))
    else:
        print('No  trained model  \'{}\''.format(config.MisakaNum))