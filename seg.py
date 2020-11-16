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


seg_model = Network()
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

class segmentation_net(nn.Module):
    def __init__(self, seg_net):
        super().__init__()
        self.seg_net = seg_net
        self.pool1 = torch.nn.AvgPool2d(kernel_size=28, stride=28)
        num_features = 4
        num_classes = 4
        self.classifier = nn.Linear(num_features, num_classes)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=2)
    def forward(self, x):
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = self.seg_net(x)
        res = lateral_map_2
        res = res.sigmoid().data
        res = (res - torch.min(res)) / (torch.max(res) - torch.min(res) + 1e-8)
        #res = torch.nn.Upsample(size=(224, 224), mode='bilinear')(res)
        res = res.view((-1,224,224))
        res = self.pool1(res)
        res = torch.flatten(res, 1)
        out = self.fc2(res)
        return out


model = segmentation_net(seg_model).cuda()

modelname = 'seg'


# train

# TODO Hyperparameters
bs = batchsize
learning_rate = 0.003
stepsize = 1
total_epoch = 5
votenum = 1

import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []

for param in model.seg_net.parameters():
    param.requires_grad = False
parameters = []
parameters.extend(model.fc2.parameters())
optimizer = optim.Adam(parameters, lr=learning_rate)

scheduler = StepLR(optimizer, step_size=stepsize, gamma=0.5)

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
        torch.save(model.state_dict(), "model_saved/SEG_lr={}_step={}_epoch={}_f1={}_acc={}_auc={}.pt".format(
            learning_rate, stepsize, '0'+str(epoch) if epoch < 10 else epoch, round(F1,4), round(acc,4), round(AUC,4)))

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



