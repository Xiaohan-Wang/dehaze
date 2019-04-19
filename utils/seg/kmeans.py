# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:52:45 2019

@author: Administrator
"""

import numpy as np
from PIL import Image as image
#加载PIL包，用于加载创建图片
from sklearn.cluster import  KMeans#加载Kmeans算法
import os
from PIL import Image

def loadData(filePath):
    f = open(filePath,'rb') #以二进制形式打开文件
    data= []
    img =image.open(f)#以列表形式返回图片像素值
    m,n =img.size     #获得图片大小
    for i in range(m):
        for j in range(n):
            #将每个像素点RGB颜色处理到0-1范围内
            x,y,z =img.getpixel((i,j))
            #将颜色值存入data内
            data.append([x/255.0,y/255.0,z/255.0])
    f.close()
    #以矩阵的形式返回data，以及图片大小
    return np.mat(data),m,n

image_path = '/home/ws/datasets/ITS(training set)/hazy' 
for fn in os.listdir(image_path):
        if fn.endswith('.png'):
            fd = os.path.join(image_path,fn)  
            imgData,row,col =loadData(fd)#加载数据

            km=KMeans(n_clusters=3)
            #聚类获得每个像素所属的类别
            label =km.fit_predict(imgData)
            label=label.reshape([row,col])
            #创建一张新的灰度图以保存聚类后的结果
            pic_new = image.new("L",(row,col))
            #根据类别向图片中添加灰度值
            for i in range(row):
                for j in range(col):
                    pic_new.putpixel((i,j),int(255/(label[i][j]+1)))
            #保存图像
            pic_new.save('/home/ws/datasets/ITS(training set)/kmeans/'+fn,'png')
