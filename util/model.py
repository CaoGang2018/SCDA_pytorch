import torch as t
import torch.nn as nn
from torchvision import models 
import torch.nn.functional as F

class pool_model(nn.Module):
    def __init__(self):
        super(pool_model, self).__init__()
        # self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
       

    def ave_pool(self, x, cc):
        b, c, h, w = x.shape
        men_pool = t.zeros(b, c)
        for i in range(b):
            count = 0
            tmp = x[i,:, 0, 0]
            # exit()
            # print(tmp.shape)
            for m in range(h):
                for n in range(w):
                    if cc[i][m][n]:
                        tmp += x[i,:, m, n]
                        count += 1
            if count == 0:
                men_pool[i] = tmp
            else:
                men_pool[i] = tmp / count
            # print(count)
        return men_pool

    def forward(self, x, cc):
        
        tmp1 = self.ave_pool(x, cc)
        tmp2 = self.pool2(x)
        b, c, _, _ = tmp2.shape
        # print(tmp1.shape)
        # print(tmp2.shape)

        return t.cat((tmp1, tmp2.reshape(b, c)), dim=1).reshape(b, -1)

# x = t.ones(2, 512, 7, 7)
# model = pool_model()
# print(model(x).shape)

# class Net(nn.modules):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.basenet = models.vgg16(pretrained=True).features
#         # print(self.basenet)

#     def forward(self, x):
#         x = self.basenet(x)
#         return x
# net1 = models.vgg16(pretrained=False).features[:-3]
# net2 = models.vgg16(pretrained=False).features
# x = t.ones(2, 3, 224, 224)
# print(net1(x).shape)
# x1 = F.interpolate(net2(x), size=net1(x).shape[-2:], mode='nearest')
# print(x1.shape)