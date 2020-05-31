from torch.utils.data import DataLoader
from data.CUB_200 import get_dataset, input_transform, input_transform2
import SCDA
from torchvision import models 
import pandas as pd
import torch
import torch.nn.functional as F
import csv
import numpy as np
from util.model import pool_model
import random


def retrieve(q, data, num=5):
    distances = np.sum(np.square((data - q)), axis=-1)
    # distances = np.sum(cosine_similarity(q, data)), axis=-1)
    indices = distances.argsort(axis=0)[:num]
    return indices


def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)


net1 = models.vgg16(pretrained=True).features[:-3]
net2 = models.vgg16(pretrained=True).features
dataset_or = get_dataset()
# dataset_filp = get_dataset(transform=input_transform2)
data_or = DataLoader(dataset_or, batch_size=10, shuffle=True,num_workers=0)
# data_flip = DataLoader(dataset_filp, batch_size=20, shuffle=True,num_workers=0)
net1.eval()
net2.eval()
result = []
label_re = []
max_ave_pool = pool_model()
# out = open('feat.csv', 'a', newline='')
# csv_write = csv.writer(out, dialect='excel')
# csv_write.writerow(['features', 'label'])
for ii, (img1, img2, label) in enumerate(data_or):
    # if ii == 2:
        # exit()
    input1 = img1
    feat_re = net1(input1)
    feat_po = net2(input1)
    input2 = img2
    feat_flip_re = net1(input2)
    feat_flip_po = net2(input2)
    m, _, h1, w1 = feat_po.shape
    m, _, h2, w2 = feat_re.shape
    label = label.detach().numpy()
    f_po = torch.zeros(feat_po.shape)
    f_re = torch.zeros(feat_re.shape)
    filp_po = torch.zeros(feat_flip_po.shape)
    filp_re = torch.zeros(feat_flip_re.shape)
    cc8 = np.zeros((m, h1, w1)) #31
    cc7 = np.zeros((m, h1, w1)) # 31
    cc6 = np.zeros((m, h2, w2))# 28
    cc5 = np.zeros((m, h2, w2))# 28
    

    for i in range(m):
        # f_re[i] = SCDA.select_aggregate(feat_re[i].detach().numpy())
        f_po[i] = SCDA.select_aggregate(feat_po[i])[0] # 31a
        cc2 = SCDA.select_aggregate(feat_po[i])[1]
        cc8[i] = cc2
        f_re[i] = SCDA.select_aggregate_and(feat_re[i], cc2)[0] # 28a
        cc3 = SCDA.select_aggregate_and(feat_re[i], cc2)[1]
        cc6[i] = cc3

        
        filp_po[i] = SCDA.select_aggregate(feat_flip_po[i])[0] # 31b
        cc2_f = SCDA.select_aggregate(feat_flip_po[i])[1]
        cc7[i] = cc2_f
        filp_re[i] = SCDA.select_aggregate_and(feat_flip_re[i], cc2_f)[0]
        cc3_f = SCDA.select_aggregate_and(feat_flip_re[i], cc2_f)[1] # 28b
        cc5[i] = cc3_f
        # print(f.shape)
        # l = int(lable[i])
        # csv_write.writerow([f, l])
        # del f, l
        # result.append((f, l))
    # print(f_po.shape)
    l31a = max_ave_pool(f_po, cc8)
    # exit()
    l28a = max_ave_pool(f_re, cc6)

    l31b = max_ave_pool(filp_po, cc7)
    l28b = max_ave_pool(filp_re, cc5)
    # print(f_po.reshape(1, -1))
    # print(f_re)
    # print(filp_po)
    # print(filp_re)

    feat = torch.cat((l31a, 0.5*l28a, l31b, 0.5*l28b), dim=1).reshape(m, -1).detach().numpy()
    # print(feat.shape)

    for i in range(m):
        # print(feat[i])
        # print(feat[i].reshape(1,-1).shape)
        result.append(feat[i].tolist())
        label_re.append(int(label[i]))
        # exit()
        
        # az = feat[i].tolist()
        # print(az)
        # csv_write.writerow([az, int(label[i])])
 
    print('batch {}/{} complete'.format(ii+1, len(data_or)))

print("test...")

feats = np.vstack(result)
# print(feats.shape)
label = np.vstack(label_re)

x = [int(i) for i in range(feats.shape[0])]
random.shuffle(x)
feats = feats[x]
labels = label[x]
# labels = [i[0] for i in label]
test_data = feats[:int(0.5*len(labels))]
train_data = feats[int(0.5*len(labels)):]
test_label = labels[:int(0.5*len(labels))]
train_label = labels[int(0.5*len(labels)):]

top1 = 0
top5 = 0
# ap = 0.0
for i in range(len(test_label)):
    # print(test_data[i].shape)
    # exit()
    inds = retrieve(test_data[i].reshape(1,-1), train_data, num=len(train_data))
    # print(inds.tolist())
    # print(inds)
    labels = train_label[inds]
    if labels[0] == test_label[i]:
        top1 +=1
    if test_label[i] in labels[:5]:
        top5 +=1
    # ap += compute_ap(inds.tolist(), get_gt(test_label[i], train_labels))
    print("Query[%d / %d] complete: top1: %.4f, top5: %.4f"%(i, len(test_label), top1/(i+1), top5/(i+1)))

print("top1 acc: %.4f, top5 acc %.4f"%(top1/len(test_label), top5/len(test_label)))

# avg.train_data_L31a
# maxi.train_data_L31a
# ratio.*avg.train_data_L28a
# ratio.*maxi.train_data_L28a 
# avg.train_data_L31b 
# maxi.train_data_L31b 
# ratio.*avg.train_data_L28b 
# ratio.*maxi.train_data_L28b
