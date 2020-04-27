#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# 用于生成PASCAL VOC格式的训练和测试txt
# Author：jefby
# Email: jef199006@gmail.com

import os
import random
import glob

# trainval数据集占所有数据的比例
trainval_percent = 0.9
# train数据集占trainval数据的比例
train_percent = 1

xmlfilepath = 'data/VOC/Annotations'
txtsavepath = 'data/VOC/ImageSets/Main'
total_xml = glob.glob(os.path.join(xmlfilepath, '*.xml'))

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)
ftrainval = open('data/VOC/ImageSets/Main/trainval.txt', 'w')
ftest = open('data/VOC/ImageSets/Main/test.txt', 'w')
ftrain = open('data/VOC/ImageSets/Main/train.txt', 'w')
fval = open('data/VOC/ImageSets/Main/val.txt', 'w')

for i in list:
    name=os.path.basename(total_xml[i])[:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()