# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



def watershed_seg(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 2)
    edge = cv2.Canny(blur, 80, 150)

    
    #find contour
    image, contours,hierarchy=cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    marks = np.zeros((h, w), np.int32)
    
    for index in range(len(contours)):
        cv2.drawContours(marks, contours, index, index + 1, 1, 8, hierarchy)    

    marks = cv2.watershed(img, marks) 
    
    marks = marks.reshape(h*w)
    
    index = np.array(np.where(marks == -1))
    index1 = np.where(index + w + 1 < h*w)
    marks[index[index1]] = marks[index[index1] + w + 1]
    
    index = np.array(np.where(marks == -1))
    index1 = np.where(index + w - 1 < h*w)
    marks[index[index1]] = marks[index[index1] + w - 1]
    
    index = np.array(np.where(marks == -1))
    index1 = np.where(index - w + 1 >= 0)
    marks[index[index1]] = marks[index[index1] - w + 1]
    
    index = np.array(np.where(marks == -1))
    index1 = np.where(index - w - 1 >= 0)
    marks[index[index1]] = marks[index[index1] - w - 1]
    
    marks = marks.reshape(h, w)    
    return marks
    

if __name__ == '__main__':
    marks_path = '/home/ws/Desktop/watershed'
    os.makedirs(marks_path, exist_ok=True)
    input_path = '/home/ws/datasets/ITS(training set)/hazy'
    for file in os.listdir(input_path):
        img = cv2.imread(input_path + '/' + file)
#        edge = cv2.imread('/home/ws/Desktop/coh/ITS' + '/' + file, 0)
        marks = watershed_seg(img)
        plt.imsave(marks_path + '/' + file, marks)
        
  
