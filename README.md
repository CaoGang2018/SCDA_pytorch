# SCDA_pytorch
A pytorch implementation of the "Selective Convolutional Descriptor Aggregation" algorithm

## NOTE
 cpu-only version

 no in [1]
 ```
 train_data_L31a(i,:) = train_data_L31a(i,:) ./ norm(train_data_L31a(i,:));
 ```


[largestConnectComponent](https://blog.csdn.net/xuyangcao123/article/details/81023732)

## Details
install requirements


```
  pip install -r requirements.txt;
```

On CUB and split dataset in `CUB_200.py`.

random split CUB-200-2011 results:
||top1|top5
---|:--:|---:
CUB|0.546|0.794



>[1] Wei X S , Luo J H , Wu J , et al. Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval[J]. IEEE Transactions on Image Processing, 2017, 26(6):2868-2881.