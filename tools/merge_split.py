# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022
import os
import random
from glob import glob
import shutil


labels = glob(r'D:\TMT Uniform\datasets\*\*.txt')  #  glob(r'D:\Dataset\TMT_uniform\datasets\*\*.txt') #+ glob(r'datasets\*\*.txt')
random.shuffle(labels)
print(len(labels))
pivot = len(labels) * 0.8
for i, label in enumerate(labels):
    img_path = label.replace('.txt', '.jpg')
    if not os.path.exists(img_path):
        continue
    if i < pivot:
        shutil.copy(label, r'D:\Dataset\TMT_uniform\train')
        shutil.copy(img_path, r'D:\Dataset\TMT_uniform\train')
    else:
        shutil.copy(label, r'D:\Dataset\TMT_uniform\val')
        shutil.copy(img_path, r'D:\Dataset\TMT_uniform\val')