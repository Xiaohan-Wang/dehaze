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
    image, contours,hierarchy=cv2.findContours(edge, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    marks = np.zeros((h, w), np.int32)
    
    for index in range(len(contours)):
        cv2.drawContours(marks, contours, index, index + 1, 1, 8, hierarchy)
   
    marks = cv2.watershed(img, marks)
    
    return marks
    

if __name__ == '__main__':
    marks_path = '/home/ws/datasets/ITS(training set)/seg'
    os.makedirs(marks_path, exist_ok=True)
    input_path = '/home/ws/datasets/ITS(training set)/hazy'
    for file in os.listdir(input_path):
        img = cv2.imread(input_path + '/' + file)
        marks = watershed_seg(img)
        plt.imsave(marks_path + '/' + file, marks, cmap='Greys_r')
        
  
