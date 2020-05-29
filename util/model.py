import torch as t
import torch.nn as nn
from torchvision import models 
import torch.nn.functional as F

class pool_model(nn.Module):
    def __init__(self):
        super(pool_model, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        tmp1 = self.pool1(x)
        tmp2 = self.pool2(x)
        b, c, _, _ = tmp1.shape
        # print(tmp1.shape)

        return t.cat((tmp1, tmp2), dim=1).reshape(b, -1)

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