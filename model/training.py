import torch
from torch import nn, optim
from model.modelseting import *
from tqdm import tqdm
from data.DataSet import getDataFromFileAboutClass
import os, shutil
import copy
from scipy.spatial.distance import cdist


def finetuning(config, teacher, student, train_data, test_data, log):

    end_epoch = config.epoch_num
    LR = config.lr

    log.printInfo("teacher")
    log.printInfo(teacher)
    log.printInfo("student")
    log.printInfo(student)
    teacher.eval()
    w_lambda = 16

    for epoch in tqdm(range(0, end_epoch), ncols=50, leave=True):
        if epoch%30 == 0:
            LR = LR*0.1
            str = 'LR = %f \n' % (LR)
            log.printInfo(str)
            params = [
                {'params': student.module.backbone.layer1.parameters(), 'lr': LR * 0.1},
                {'params': student.module.backbone.layer2.parameters(), 'lr': LR * 0.1},
                {'params': student.module.backbone.layer3.parameters(), 'lr': LR * 0.2},
                {'params': student.module.backbone.layer4.parameters(), 'lr': LR * 0.5},
                {'params': student.module.fc.parameters()}, ]
            optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)

        student.train()

        total_loss1 = torch.tensor([0]).cuda()
        total_loss2 = torch.tensor([0]).cuda()
        total_loss3 = torch.tensor([0]).cuda()
        total_sampleNum = 0
        total_correct = 0
        for batchidx, (x, label) in enumerate(train_data):
            x, label = x.cuda(), label.cuda()
            oldIdx = torch.where(label < config.preClassNum)[0]
            newIdx = torch.where(label >= config.preClassNum)[0]

            with torch.no_grad():
                teacher_out, teacher_xout = teacher(x)
            student_out, student_xout = student(x)

            if oldIdx.shape != torch.Tensor([]).shape and newIdx.shape != torch.Tensor([]).shape:
                loss_old = cosMaxLossFunc(student_out[oldIdx], label[oldIdx], config.classNum)
                loss_new = cosMaxLossFunc(student_out[newIdx], label[newIdx], config.classNum)
                loss1 = (loss_old*oldIdx.shape[0]*w_lambda+loss_new*newIdx.shape[0])/(oldIdx.shape[0]*w_lambda+newIdx.shape[0])
            elif newIdx.shape != torch.Tensor([]).shape:
                loss1 = cosMaxLossFunc(student_out[newIdx], label[newIdx], config.classNum)
            else:
                loss1 = cosMaxLossFunc(student_out[oldIdx], label[oldIdx], config.classNum)

            if config.step > 0:
                loss2 = rooLossFunc(teacher_out, student_out, config.preClassNum)
                loss3 = feature_distll_loss(teacher_xout, student_xout)
            else:
                loss2 = torch.Tensor([0]).cuda()
                loss3 = torch.Tensor([0]).cuda()


            loss = loss1 + loss2 + loss3
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # loss
                total_loss1 = total_loss1 + loss1.detach()
                total_loss2 = total_loss2 + loss2.detach()
                total_loss3 = total_loss3 + loss3.detach()

                pred = student_out.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_sampleNum += x.shape[0]

        with torch.no_grad():
            train_acc = total_correct/total_sampleNum
            str = '%d \n:loss1=%.5f, loss2=%.5f, loss3=%.5f, trainAcc=%.5f' %(epoch, total_loss1.item(), total_loss2.item(), total_loss3.item(), train_acc)
            # str = '%d \n:loss1=%.5f, loss2=%.5f, trainAcc=%.5f' %(epoch, total_loss1.item(), total_loss2.item(), train_acc)

        teacher.eval()
        student.eval()
        with torch.no_grad():
            # test
            total_loss1 = torch.tensor([0]).cuda()
            total_loss2 = torch.tensor([0]).cuda()
            total_loss3 = torch.tensor([0]).cuda()
            total_sampleNum = 0
            total_correct = 0
            for x, label in test_data:
                x, label = x.cuda(), label.cuda()
                teacher_out, teacher_xout = teacher(x)
                student_out, student_xout = student(x)
                loss1 = cosMaxLossFunc(student_out, label, config.classNum)
                if config.step > 0:
                    loss2 = rooLossFunc(teacher_out, student_out, config.preClassNum)
                    loss3 = feature_distll_loss(teacher_xout,student_xout)
                else:
                    loss2 = torch.Tensor([0]).cuda()
                    loss3 = torch.Tensor([0]).cuda()
                # loss
                total_loss1 = total_loss1 + loss1.detach()
                total_loss2 = total_loss2 + loss2.detach()
                total_loss3 = total_loss3 + loss3.detach()
                pred = student_out.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_sampleNum += x.shape[0]

            test_acc = total_correct/total_sampleNum
            str += ', loss1=%.5f, loss2=%.5f, loss3=%.5f, validAcc=%.5f' % (total_loss1.item(), total_loss2.item(), total_loss3.item(), test_acc)
            # str += ', loss1=%.5f, loss2=%.5f, validAcc=%.5f' % (total_loss1.item(), total_loss2.item(), test_acc)
            log.printInfo(str)
    model_save(student, config.model_path, config.USE_MULTI_GPU)



def cosMaxLossFunc(cosine, label, classNum):
    label_onehot = nn.functional.one_hot(label, num_classes=classNum)

    molecule = torch.sum(torch.mul(cosine, label_onehot), dim=1, keepdim=True)
    denominator = torch.sum(torch.pow(cosine, 2), dim=1, keepdim=True)
    denominator = torch.sqrt(denominator)

    loss = torch.div(molecule, denominator)
    loss = torch.pow(1-loss, 2)

    loss = torch.sum(loss)/cosine.shape[0]

    return loss

def rooLossFunc(teacherOut, studentOut, classNum):
    teacherOut = teacherOut[:, :classNum]
    studentOut = studentOut[:, :classNum]

    loss = nn.MSELoss()(studentOut, teacherOut)
    return  loss
def feature_distll_loss(teacher_feature, student_feature):
    loss = nn.CosineEmbeddingLoss()(teacher_feature, student_feature, torch.ones(student_feature.shape[0]).cuda())
    return loss
def rofLossFunc(teacherXout, studentXout):
    loss = nn.MSELoss()(studentXout, teacherXout)
    return loss

def stepTest(config, log):
    model_path = "./model/" + config.TrainId + '-' + str(config.step) + ".pth"
    model = model_load(model_path)
    if config.USE_MULTI_GPU:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    for i in range(0, config.step+1):
        LS = i*config.incr_cls
        LE = (i+1)*config.incr_cls
        if(config.allClassNum-LE<config.incr_cls):
            LE = config.allClassNum
        test_dataloader = getDataFromFileAboutClass(config.testDir, config.test_transformer,
                                                    LS, LE, config.batchsz)
        model.eval()
        with torch.no_grad():
            total_sampleNum = 0
            total_correct = 0
            for x, label, _ in test_dataloader:
                x, label = x.cuda(), label.cuda()

                out, xout = model(x)

                pred = out.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_sampleNum += x.shape[0]

            test_acc = total_correct / total_sampleNum
        logstr = 'step=%d, testAcc=%.5f' % (i, test_acc)
        # print(logstr)
        log.printInfo(logstr)

def get_featureMean(config, test_data):
    model_path = "./model/" + config.MisakaNum + '-' + str(config.step) + ".pth"
    model = model_load(model_path)
    if config.USE_MULTI_GPU:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    LE = config.classNum
    class_means = np.zeros((LE, 2048))
    for c in range(LE):
        train_dataloader = getDataFromFileAboutClass(config.inversionDir, config.test_transformer,
                                                     c, c + 1, config.batchsz)
        model.eval()
        with torch.no_grad():
            total_sampleNum = 0
            featureMean = torch.zeros([1, 2048]).cuda()
            for x, label, filePath in train_dataloader:
                x, label = x.cuda(), label.cuda()
                out, xout = model(x)
                featureMean = featureMean + torch.sum(xout, dim=0)
                total_sampleNum += x.shape[0]
            featureMean = featureMean / total_sampleNum

            normFeatureMean = torch.div(featureMean, torch.norm(featureMean, p=2, dim=1, keepdim=True))
            #normFeatureMean to numpy
            class_means[c] = normFeatureMean.cpu().numpy()
    xout ,label_true= [], []
    test_sampleNum = 0
    model.eval()
    with torch.no_grad():
        for x, label in test_data:
            x, label = x.cuda(), label.cuda()
            test_sampleNum += x.shape[0]
            _, _xout = model(x)
            xout.append(_xout)
            label_true.append(label)
        xout = torch.cat(xout, dim=0).cpu().numpy()
        label_true = torch.cat(label_true).cpu().numpy()
        xout_norm = (xout.T / (np.linalg.norm(xout.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, xout_norm, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
        pred = np.argsort(scores, axis=1)[:, : 1]

        correct = (pred.squeeze() == label_true).sum().item()
        test_acc = correct / test_sampleNum
        printstr = 'NMEACC-tok1=%.5f' % (test_acc)
        print(printstr)


EPSILON = 1e-8


def select_img(config):
    model_path = "./model/" + config.MisakaNum + '-' + str(config.step) + ".pth"
    model = model_load(model_path)
    if config.USE_MULTI_GPU:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    LE = config.classNum
    LS = config.incr_cls*config.step
    for c in range(LS, LE):
        train_dataloader = getDataFromFileAboutClass(config.trainDir, config.test_transformer,
                                                    c, c+1, config.batchsz)
        model.eval()
        with torch.no_grad():
            total_sampleNum = 0
            featureMean = torch.zeros([1, 2048]).cuda()
            features = torch.Tensor([]).cuda()
            path = []
            for x, label, filePath in train_dataloader:
                x, label = x.cuda(), label.cuda()

                out, xout = model(x)

                featureMean = featureMean+torch.sum(xout, dim=0)
                total_sampleNum += x.shape[0]

                features = torch.cat((features, xout), dim=0)
                filePath = list(filePath)
                path = path+filePath
            featureMean = featureMean / total_sampleNum
            normFeatures = torch.div(features, torch.norm(features, p=2, dim=1, keepdim=True))
            normFeatureMean = torch.div(featureMean, torch.norm(featureMean, p=2, dim=1, keepdim=True))
            dist = 1-torch.mm(normFeatures, normFeatureMean.t())
            dist = dist.t()
            dist = dist.squeeze(0)
            dist = dist.cpu().numpy()
            top_idx = np.argpartition(dist, config.reserveNum)[:config.reserveNum]
            save_imgs = [path[i] for i in top_idx]
            print(save_imgs)
            for src in save_imgs:
                dst = src.replace('train', 'reserve1')
                dir = os.path.dirname(src)
                dir = dir.replace('train', 'reserve1')
                if not os.path.exists(dir):
                    os.makedirs(dir)
                shutil.copy(src, dst)
