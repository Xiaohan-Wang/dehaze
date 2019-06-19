# -*- coding: utf-8 -*-

import cv2
from config import Config
import glob

def refine_trans(imgs):
    opt = Config()
    for img in imgs:
        trans = cv2.imread(img)
        hazy = cv2.imread("/home/ws/Desktop/benchmark/" + img.split("/")[-1].split(".")[0] + ".png")
        hazy = hazy[:,:,::-1]
        refined_trans = cv2.ximgproc.guidedFilter(guide=hazy, src=trans, radius=60, eps=10)
        cv2.imwrite(opt.refined_trans + '/' + img.split("/")[-1], refined_trans)
    
    
if __name__ ==  "__main__":
    imgs = glob.glob('/home/ws/xh/dehaze_spyder/trans/*')
    refine_trans(imgs)
    print("Done!")