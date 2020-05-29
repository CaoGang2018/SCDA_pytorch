from torch.utils.data import DataLoader
from data.CUB_200 import get_dataset, input_transform, input_transform2
import SCDA
from torchvision import models 
import torch
import torch.nn.functional as F
import csv
import numpy as np
from util.model import pool_model


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
max_ave_pool = pool_model()
out = open('feat.csv', 'a', newline='')
csv_write = csv.writer(out, dialect='excel')
csv_write.writerow(['features', 'label'])
for ii, (img1, img2, label) in enumerate(data_or):
    input1 = img1
    feat_re = net1(input1)
    feat_po = net2(input1)
    input2 = img2
    feat_flip_re = net1(input2)
    feat_flip_po = net2(input2)
    m, _, _, _ = feat_re.shape
    label = label.detach().numpy()
    f_po = torch.zeros(feat_po.shape)
    f_re = torch.zeros(feat_re.shape)
    filp_po = torch.zeros(feat_flip_po.shape)
    filp_re = torch.zeros(feat_flip_re.shape)
    for i in range(m):
        # f_re[i] = SCDA.select_aggregate(feat_re[i].detach().numpy())
        f_po[i] = SCDA.select_aggregate(feat_po[i])[0] # 31a
        cc2 = SCDA.select_aggregate(feat_po[i])[1]
        f_re[i] = SCDA.select_aggregate_and(feat_re[i], cc2) # 28a

        
        filp_po[i] = SCDA.select_aggregate(feat_flip_po[i])[0] # 31b
        cc2_f = SCDA.select_aggregate(feat_flip_po[i])[1]
        filp_re[i] = SCDA.select_aggregate_and(feat_flip_re[i], cc2_f) # 28b
        # print(f.shape)
        # l = int(lable[i])
        # csv_write.writerow([f, l])
        # del f, l
        # result.append((f, l))
    # print(f_po.shape)
    l31a = max_ave_pool(f_po)
    l28a = max_ave_pool(f_re)

    l31b = max_ave_pool(filp_po)
    l28b = max_ave_pool(filp_re)

    feat = torch.cat((l31a, 0.5*l28a, l31b, 0.5*l28b), dim=1).reshape(m, -1).detach().numpy()

    for i in range(m):
        csv_write.writerow([feat[i], int(label[i])])
 
    print('batch {} complete'.format(ii))




# avg.train_data_L31a
# maxi.train_data_L31a
# ratio.*avg.train_data_L28a
# ratio.*maxi.train_data_L28a 
# avg.train_data_L31b 
# maxi.train_data_L31b 
# ratio.*avg.train_data_L28b 
# ratio.*maxi.train_data_L28b
