import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join
from scipy.io import loadmat

from collections import namedtuple
from PIL import Image



rootdir = 'F:\Paper\SCDA\SCDA_pytorch_CODE\data\images'
dbStruc = namedtuple('dbStruct', ['name', 'class1'])

'''
path = 'imdb.mat'
mat = loadmat(path)
matStruct = mat['images'].item()
dbImage = matStruct[1]
print(dbImage.shape)
'''


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['images'].item()
    dbImage = matStruct[1][0]
    dbclass = matStruct[3]

    return dbStruc(dbImage, dbclass)


def input_transform():
    return transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 将Tensor正则化
                              std=[0.229, 0.224, 0.225]),
    ])

def input_transform2():
    return transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 将Tensor正则化
                              std=[0.229, 0.224, 0.225]),
    ])


def get_dataset():
    structFile = join(rootdir, 'imdb.mat')
    return WholeDataset(structFile)
img_dir = 'F:\Datesets\CUB_200\CUB_200_2011\CUB_200_2011\images'

class WholeDataset(data.Dataset):
    def __init__(self, structFile):
        super(WholeDataset, self).__init__()
        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(img_dir, dbim[0] + '.jpg') for dbim in self.dbStruct[0]]
        self.classes = self.dbStruct[1][0]
        self.input_transform1 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 将Tensor正则化
                              std=[0.229, 0.224, 0.225]),
        ])
        self.input_transform2 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1), # flip
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 将Tensor正则化
                              std=[0.229, 0.224, 0.225]),
        ])

    # def input_transform(self):
    #     return transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 将Tensor正则化
    #                           std=[0.229, 0.224, 0.225]),
    #     ])

    # def input_transform2(self):
    #     return transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.RandomHorizontalFlip(p=1), # flip
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 将Tensor正则化
    #                           std=[0.229, 0.224, 0.225]),
    #     ])


    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.convert('RGB')
        label = self.classes[index]

        img1 = self.input_transform1(img)
        img2 = self.input_transform2(img)

        return img1, img2, label

    def __len__(self):
        return len(self.images)




