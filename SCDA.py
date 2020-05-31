import torch as t
import torchvision.models as model
import torch.nn.functional as F
from util.largestConnectComponent import largestConnectComponent

"""
feat_map = t.ones(5, 2, 4)
print(feat_map)
A = feat_map.sum(dim=0)
print(A)
a = A.mean(dim=[0, 1])
print(float(a))
"""

def select_aggregate(feat_map):
    A = t.sum(feat_map, dim=[0])
    a = t.mean(A, dim=[0, 1]).float()
    tmp = t.ones(A.shape)
    tmp[A<a] = 0
    tmp_out = t.zeros(feat_map.shape)
    cc = largestConnectComponent(tmp)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if cc[i, j]:
                tmp_out[:, i, j] = feat_map[:, i, j]
    return tmp_out, cc

def select_aggregate_and(feat_map, cc2): # relu5_2
    A = t.sum(feat_map, dim=[0])
    a = t.mean(A, dim=[0, 1]).float()
    tmp = t.ones(A.shape)
    tmp[A<a] = 0
    tmp_out = t.zeros(feat_map.shape)
    cc = largestConnectComponent(tmp)
    p, q = cc2.shape
    tm = t.zeros(cc2.shape)
    for i in range(p):
        for j in range(q):
            if cc2[i, j]:
                tm[i, j] = 1
    # return tmp_out
    cc_and = F.interpolate(t.from_numpy(tm.reshape(1, 1, p, q).numpy()), size=cc.shape, mode='nearest').reshape(cc.shape)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if cc[i, j] and cc_and[i, j] == 1:
                tmp_out[:, i, j] = feat_map[:, i, j]
    return tmp_out, cc_and

# x = t.randn(3, 4, 4)
# # print(type(x))
# print(x)
# sx = select_aggregate(x)
# print(sx)
# print(cc)
# tx = t.zeros(x.shape)
# cc = largestConnectComponent(sx)
# for i in range(sx.shape[0]):
    # for j in range(sx.shape[1]):
        # if cc[i, j]:
            # tx[:, i, j] = x[:, i, j]
# print(tx)


