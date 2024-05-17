import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image

#官方给出的python3解压数据文件函数，返回数据字典
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
#
#
#
# #训练集有五个批次，每个批次10000个图片，测试集有10000张图片
# def cifar10_img(file_dir):
#     loc_1 = './data/Cifar10Image/train/'
#     loc_2 = './data/Cifar10Image/test/'
#
#     # 判断文件夹是否存在，不存在的话创建文件夹
#     if os.path.exists(loc_1) == False:
#         os.mkdir(loc_1)
#     if os.path.exists(loc_2) == False:
#         os.mkdir(loc_2)
#
#     for i in range(1,6):
#         data_name = file_dir + '/'+'data_batch_'+ str(i)
#         data_dict = unpickle(data_name)
#         print(data_name + ' is processing')
#
#         for j in range(10000):
#             img = np.reshape(data_dict[b'data'][j],(3,32,32))
#             img = np.transpose(img,(1,2,0))
#             #通道顺序为RGB
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#             #要改成不同的形式的文件只需要将文件后缀修改即可
#             if os.path.exists(loc_1 + str(data_dict[b'labels'][j])) == False:
#                 os.mkdir(loc_1 + str(data_dict[b'labels'][j]))
#             img_name = loc_1 + str(data_dict[b'labels'][j])+'/'+str((i)*10000 + j) +'.bmp'
#             cv2.imwrite(img_name,img)
#
#         print(data_name + ' is done')
#
#
#     test_data_name = file_dir + '/test_batch'
#     print(test_data_name + ' is processing')
#     test_dict = unpickle(test_data_name)
#
#     for m in range(10000):
#         img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
#         img = np.transpose(img, (1, 2, 0))
#         # 通道顺序为RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # 要改成不同的形式的文件只需要将文件后缀修改即可
#         if os.path.exists(loc_2 + str(test_dict[b'labels'][m])) == False:
#             os.mkdir(loc_2 + str(test_dict[b'labels'][m]))
#         img_name = loc_2 + str(test_dict[b'labels'][m])+ '/' + str(10000 + m) + '.bmp'
#         cv2.imwrite(img_name, img)
#     print(test_data_name + ' is done')
#     print('Finish transforming to image')
#
#
def ToImageMain():
    file_dir = "./../Dataset/cifar10"
    imagePath = "./../Dataset/cifar10Transforms"
    # for i in range(0, 100):
    #     if os.path.exists(imagePath+"/train/"+str(i)):
    #         print(imagePath+"/train/"+str(i))
    #     else:
    #         os.mkdir(imagePath+"/train/"+str(i))
    #
    #     if os.path.exists(imagePath+"/test/"+str(i)):
    #         print(imagePath+"/test/"+str(i))
    #     else:
    #         os.mkdir(imagePath+"/test/"+str(i))

    tensorToImage(file_dir, imagePath)


def tensorToImage(datasetPath, imagePath):
    # sizetransformer = transforms.Compose([
    #     transforms.Resize([32, 32]),
    #     transforms.ToTensor()
    # ])
    color_jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)
    sizetransformer = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomCrop([224, 224], padding=4, pad_if_needed=False, fill=0, padding_mode='constant'),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([AddSaltPepperNoise(0.2)], p=0.8),
        transforms.RandomApply([AddGaussianNoise(mean=0, variance=8, amplitude=4)], p=0.3),
        transforms.ToTensor()
    ])

    unloader = transforms.ToPILImage()
    cifar_train = datasets.CIFAR10(datasetPath, True, transform=sizetransformer, download=True)
    cifar_test = datasets.CIFAR10(datasetPath, False, transform=sizetransformer, download=True)
    train_data = DataLoader(cifar_train, batch_size=1, shuffle=True)  # shuffle=True，先打乱，再取batch。
    test_data = DataLoader(cifar_test, batch_size=1, shuffle=True)

    for idx, (x, label) in enumerate(train_data):
        x, label = x.cpu(), label.cpu()
        x = x.squeeze(0)
        img = unloader(x)  # tensor转为PIL Image
        # savePath = imagePath+'/train/'+str(label.item())+'/'+str(idx)+'.bmp'
        img.show()
        # img.save(savePath)

    for idx, (x, label) in enumerate(test_data):
        x, label = x.cpu(), label.cpu()
        x = x.squeeze(0)
        img = unloader(x)  # tensor转为PIL Image
        # savePath = imagePath+'/test/'+str(label.item())+'/'+str(idx)+'.bmp'
        # img.save(savePath)




class AddSaltPepperNoise(object):

    def __init__(self, density=0.2):
        self.density = density

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255  # 避免有值超过255而反转
        img[img < 0] = 0
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img