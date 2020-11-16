import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime
import pandas as pd
import random
from torchvision.datasets import ImageFolder
import re
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
from skimage.io import imread, imsave
import skimage
from PIL import ImageFile
from PIL import Image
import sys
sys.path.append('/home/gcy/projects/COVID-CT-master/baseline methods')
import Densenet
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset
torch.cuda.empty_cache()

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

########## Mean and std are calculated from the train dataset
normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                 std=[0.33165374, 0.33165374, 0.33165374])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(90),
    # random brightness and random contrast
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

batchsize = 4


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample


if __name__ == '__main__':
    trainset = CovidCTDataset(root_dir='/home/gcy/projects/COVID-CT-master/Images-processed',
                              txt_COVID='/home/gcy/projects/COVID-CT-master/Data-split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='/home/gcy/projects/COVID-CT-master/Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir='/home/gcy/projects/COVID-CT-master/Images-processed',
                            txt_COVID='/home/gcy/projects/COVID-CT-master/Data-split/COVID/valCT_COVID.txt',
                            txt_NonCOVID='/home/gcy/projects/COVID-CT-master/Data-split/NonCOVID/valCT_NonCOVID.txt',
                            transform=val_transformer)
    testset = CovidCTDataset(root_dir='/home/gcy/projects/COVID-CT-master/Images-processed',
                             txt_COVID='/home/gcy/projects/COVID-CT-master/Data-split/COVID/testCT_COVID.txt',
                             txt_NonCOVID='/home/gcy/projects/COVID-CT-master/Data-split/NonCOVID/testCT_NonCOVID.txt',
                             transform=val_transformer)
    trainvalset = CovidCTDataset(root_dir='/home/gcy/projects/COVID-CT-master/Images-processed',
                              txt_COVID='/home/gcy/projects/COVID-CT-master/Data-split/COVID/trainvalCT_COVID.txt',
                              txt_NonCOVID='/home/gcy/projects/COVID-CT-master/Data-split/NonCOVID/trainvalCT_NonCOVID.txt',
                              transform=train_transformer)
    print("Training set size:\t", trainset.__len__())
    print("Validation set size:\t", valset.__len__())
    print("Test set size:\t\t", testset.__len__())

    #train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    #val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    #test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    # TODO merge train and val for training
    train_loader = DataLoader(trainvalset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

for batch_index, batch_samples in enumerate(train_loader):
    data, target = batch_samples['img'], batch_samples['label']
skimage.io.imshow(data[0, 1, :, :].numpy())


# training process is defined here

alpha = None
## alpha is None if mixup is not used
alpha_name = f'{alpha}'
device = 'cuda'


def train(optimizer, epoch):
    model.train()

    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        ## adjust data to meet the input dimension of model
        #         data = data[:, 0, :, :]
        #         data = data[:, None, :, :]

        # mixup
        #         data, targets_a, targets_b, lam = mixup_data(data, target, alpha, use_cuda=True)

        optimizer.zero_grad()
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())

        # mixup loss
        #         loss = mixup_criterion(criteria, output, targets_a, targets_b, lam)

        train_loss += criteria(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item() / bs))

# val process is defined here

def val(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            #             data = data[:, 0, :, :]
            #             data = data[:, None, :, :]
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            #             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            #             print(output[:,1].cpu().numpy())
            #             print((output[:,1]+output[:,0]).cpu().numpy())
            #             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist

# test process is defined here

def tenetwork(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)
    return targetlist, scorelist, predlist

'''CheXNet pretrained model'''

# class DenseNet121(nn.Module):
#     """Model modified.

#     The architecture of our model is the same as standard DenseNet121
#     except the classifier layer which has an additional sigmoid function.

#     """
#     def __init__(self, out_size):
#         super(DenseNet121, self).__init__()
#         self.densenet121 = torchvision.models.densenet121(pretrained=True)
#         num_ftrs = self.densenet121.classifier.in_features
#         self.densenet121.classifier = nn.Sequential(
#             nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.densenet121(x)
#         return x


# device = 'cuda'
# CKPT_PATH = 'model.pth.tar'
# N_CLASSES = 14

# DenseNet121 = DenseNet121(N_CLASSES).cuda()

# CKPT_PATH = './CheXNet/model.pth.tar'

# if os.path.isfile(CKPT_PATH):
#     checkpoint = torch.load(CKPT_PATH)
#     state_dict = checkpoint['state_dict']
#     remove_data_parallel = False


#     pattern = re.compile(
#         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
#     for key in list(state_dict.keys()):
#         match = pattern.match(key)
#         new_key = match.group(1) + match.group(2) if match else key
#         new_key = new_key[7:] if remove_data_parallel else new_key
#         new_key = new_key[7:]
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]


#     DenseNet121.load_state_dict(checkpoint['state_dict'])
#     print("=> loaded checkpoint")
# #     print(densenet121)
# else:
#     print("=> no checkpoint found")

# # for parma in DenseNet121.parameters():
# #         parma.requires_grad = False
# DenseNet121.densenet121.classifier._modules['0'] = nn.Linear(in_features=1024, out_features=2, bias=True)
# DenseNet121.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# # print(DenseNet121)
# model = DenseNet121.to(device)

# %%

'''DenseNet121 pretrained model from xrv'''

# class DenseNetModel(nn.Module):

#     def __init__(self):
#         """
#         Pass in parsed HyperOptArgumentParser to the model
#         :param hparams:
#         """
#         super(DenseNetModel, self).__init__()

#         self.dense_net = xrv.models.DenseNet(num_classes=2)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         logits = self.dense_net(x)
#         return logits

# model = DenseNetModel().cuda()
# modelname = 'DenseNet_medical'
# # print(model)

# %%

'''ResNet18 pretrained'''
# import torchvision.models as models
# model = models.resnet18(pretrained=True).cuda()
# modelname = 'ResNet18'

# %%

'''Dense121 pretrained'''
# import torchvision.models as models
# model = models.densenet121(pretrained=True).cuda()
# modelname = 'Dense121'
# pretrained_net = torch.load('model_backup/Dense121.pt')
# model.load_state_dict(pretrained_net)

# %%

### Dense169
import torchvision.models as models

# model = models.densenet169(pretrained=True).cuda()
# # # modelname = 'Dense169'

# """load MoCo pretrained model"""
# checkpoint = torch.load('new_data/save_model_dense/checkpoint_luna_covid_moco.pth.tar')
# # # # print(checkpoint.keys())
# # # # print(checkpoint['arch'])

# state_dict = checkpoint['state_dict']
# for key in list(state_dict.keys()):
#     if 'module.encoder_q' in key:
# #         print(key[17:])
#         new_key = key[17:]
#         state_dict[new_key] = state_dict[key]
#     del state_dict[key]
# for key in list(state_dict.keys()):
#     if  key == 'classifier.0.weight':
#         new_key = 'classifier.weight'
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]
#     if  key == 'classifier.0.bias':
#         new_key = 'classifier.bias'
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]
#     if  key == 'classifier.2.weight' or key == 'classifier.2.bias':
#         del state_dict[key]
# state_dict['classifier.weight'] = state_dict['classifier.weight'][:1000,:]
# state_dict['classifier.bias'] = state_dict['classifier.bias'][:1000]
# model.load_state_dict(checkpoint['state_dict'])

# # # print(model)

# %%

"""Load Self-Trans model"""
"""Change names and locations to the Self-Trans.pt"""



#model = models.densenet169(pretrained=True).cuda()
seg_model = Network()
# pretrained_net = torch.load('model_backup/Dense169.pt')
# pretrained_net = torch.load('model_backup/mixup/Dense169_0.6.pt')
seg_model.load_state_dict(torch.load('pretrained/Semi-Inf-Net-100.pth', map_location={'cuda:1':'cuda:0'}))
cls_model = Densenet.densenet169()
pretrained_net = torch.load('/home/gcy/projects/COVID-CT-master/baseline methods/Self-Trans/Self-Trans.pt')

cls_model.load_state_dict(pretrained_net)


class wh_net(nn.Module):
    def __init__(self, seg_net, cls_net):
        super().__init__()
        self.seg_net = seg_net
        self.cls_net = cls_net
        self.con_c = torch.nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = self.seg_net(x)
        res = lateral_map_2
        # res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data
        res = (res - torch.min(res)) / (torch.max(res) - torch.min(res) + 1e-8)
        image_mask = torch.cat((x, res), 1)
        input = self.con_c(image_mask)
        output = self.cls_net(input)
        return output

model = wh_net(seg_model, cls_model).cuda()

modelname = 'Dense169_ssl_luna_moco'

# %%

'''ResNet50 pretrained'''

# import torchvision.models as models
# model = models.resnet50(pretrained=True).cuda()

# checkpoint = torch.load('new_data/save_model/checkpoint.pth.tar')
# # print(checkpoint.keys())
# # print(checkpoint['arch'])

# state_dict = checkpoint['state_dict']
# for key in list(state_dict.keys()):
#     if 'module.encoder_q' in key:
#         print(key[17:])
#         new_key = key[17:]
#         state_dict[new_key] = state_dict[key]
#     del state_dict[key]
# for key in list(state_dict.keys()):
#     if  key == 'fc.0.weight':
#         new_key = 'fc.weight'
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]
#     if  key == 'fc.0.bias':
#         new_key = 'fc.bias'
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]
#     if  key == 'fc.2.weight' or key == 'fc.2.bias':
#         del state_dict[key]
# state_dict['fc.weight'] = state_dict['fc.weight'][:1000,:]
# state_dict['fc.bias'] = state_dict['fc.bias'][:1000]
# # print(state_dict.keys())

# # print(state_dict)
# # pattern = re.compile(
# #         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
# #     for key in list(state_dict.keys()):
# #         match = pattern.match(key)
# #         new_key = match.group(1) + match.group(2) if match else key
# #         new_key = new_key[7:] if remove_data_parallel else new_key
# #         new_key = new_key[7:]
# #         state_dict[new_key] = state_dict[key]
# #         del state_dict[key]

# # model.load_state_dict(checkpoint['state_dict'])

# # # modelname = 'ResNet50'
# modelname = 'ResNet50_ssl'

# %%

'''VGGNet pretrained'''
# import torchvision.models as models
# model = models.vgg16(pretrained=True)
# model = model.cuda()
# modelname = 'vgg16'

# %%

'''efficientNet pretrained'''

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
# model = model.cuda()
# modelname = 'efficientNet-b0'


# model = EfficientNet.from_name('efficientnet-b1').cuda()
# modelname = 'efficientNet_random'

# %%


# %%

# train

# TODO Hyperparameters
bs = batchsize
train_all = False
learning_rate = 0.000001
stepsize = 1
total_epoch = 3 * stepsize
votenum = 1

import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
if train_all == True:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    #train con c layer  cls net
    for param in model.seg_net.parameters():
        param.requires_grad = False
    parameters = []
    parameters.extend(model.con_c.parameters())
    parameters.extend(model.cls_net.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)


# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

scheduler = StepLR(optimizer, step_size=stepsize, gamma=0.1)

for epoch in range(1, total_epoch + 1):
    train(optimizer, epoch)

    # TODO trainval
    #targetlist, scorelist, predlist = val(epoch)
    targetlist, scorelist, predlist = tenetwork(epoch)
    vote_pred = np.zeros(testset.__len__()) # valset
    vote_score = np.zeros(testset.__len__()) # valset
    vote_pred = vote_pred + predlist
    vote_score = vote_score + scorelist

    if epoch % votenum == 0:
        # major vote
        vote_pred[vote_pred <= (votenum / 2)] = 0
        vote_pred[vote_pred > (votenum / 2)] = 1
        vote_score = vote_score / votenum

        #print('vote_pred', vote_pred)
        #print('targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()

        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        print('TP+FP', TP + FP)
        p = TP / (TP + FP)
        print('precision', p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall', r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1', F1)
        print('acc', acc)
        AUC = roc_auc_score(targetlist, vote_score)
        print('AUCp', roc_auc_score(targetlist, vote_pred))
        print('AUC', AUC)

        #         if epoch == total_epoch:
        # torch.save(model.state_dict(), "model_saved/{}_{}_covid_moco_covid.pt".format(modelname, alpha_name))
        torch.save(model.state_dict(), "model_saved/skinn_{}_lr={}_step={}_epoch={}_f1={}_acc={}_auc={}.pt".format(
            str("all" if train_all else "cls"), learning_rate, stepsize, '0'+str(epoch) if epoch < 10 else epoch, round(F1,4), round(acc,4), round(AUC,4)))

        print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
            epoch, r, p, F1, acc, AUC))

# test
bs = 10
import warnings

warnings.filterwarnings('ignore')

epoch = 1
r_list = []
p_list = []
acc_list = []
AUC_list = []
# TP = 0
# TN = 0
# FN = 0
# FP = 0
vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())

targetlist, scorelist, predlist = tenetwork(epoch)
#print('target', targetlist)
#print('score', scorelist)
#print('predict', predlist)
vote_pred = vote_pred + predlist
vote_score = vote_score + scorelist

TP = ((predlist == 1) & (targetlist == 1)).sum()

TN = ((predlist == 0) & (targetlist == 0)).sum()
FN = ((predlist == 0) & (targetlist == 1)).sum()
FP = ((predlist == 1) & (targetlist == 0)).sum()

print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
print('TP+FP', TP + FP)
p = TP / (TP + FP)
print('precision', p)
p = TP / (TP + FP)
r = TP / (TP + FN)
print('recall', r)
F1 = 2 * r * p / (r + p)
acc = (TP + TN) / (TP + TN + FP + FN)
print('F1', F1)
print('acc', acc)
AUC = roc_auc_score(targetlist, vote_score)
print('AUC', AUC)



