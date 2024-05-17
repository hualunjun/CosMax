from config import Config
from sys import path

path.append('./data')
path.append('./model')
path.append('./tool')

from data.DataSet import getTopClassDataFromFile
from model.modelseting import MisakaNet
from model.training import *
from log.LogCtrl import *
from model.interface import *
import os

def copyParameter(student, teacther, ratio=0.5):
    for p_student, p_teacher in zip(student.parameters(), teacther.parameters()):
        avg_p = p_student*(1-ratio)+p_teacher*ratio
        p_student.data.copy_(avg_p)

def main():
    for i in range(0, 5):
        cfg = Config(i)
        log = Misakalog(cfg)

        train_dataloader = getTopClassDataFromFile(cfg.inversionDir, cfg.trainDir, cfg.train_transformer, cfg, isEqual=False)
        test_dataloader = getTopClassDataFromFile(cfg.testDir, cfg.testDir, cfg.test_transformer, cfg)

        if cfg.step== 0:
            teacher = MisakaNet(cfg)
            student = MisakaNet(cfg)
        else:
            teacher_model_path = "./model/" + cfg.TrainId + '-' + str(cfg.step-1) + ".pth"
            teacher = model_load(teacher_model_path)
            if teacher.fc.preWeight is None:
                preweight = teacher.fc.currentWeight.data
            else:
                preweight = torch.cat((teacher.fc.preWeight.data, teacher.fc.currentWeight.data), 1)
            student = MisakaNet(cfg, preweight)
        #Model Inheritance
        student = student.cpu()
        teacher = teacher.cpu()
        copyParameter(student.backbone.layer3, teacher.backbone.layer3, ratio=0.8)
        copyParameter(student.backbone.layer4, teacher.backbone.layer4, ratio=0.8)
        if cfg.USE_MULTI_GPU:
            teacher = torch.nn.DataParallel(teacher).cuda()  # muti_gpu
            student = torch.nn.DataParallel(student).cuda()
        else:
            teacher = teacher.cuda()  # single_gpu
            student = student.cuda()


        finetuning(cfg, teacher, student, train_dataloader, test_dataloader, log)
        stepTest(cfg, log)




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'   #set gpu id
    main()


