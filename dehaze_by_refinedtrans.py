from DehazeNet import DehazeNet
from PIL import Image
from config import Config
import glob
import torch
import torchvision
from utils.airlight import estimate_airlight
import utils.save_load as sl
from utils.seg.watershed import watershed_seg as seg
import cv2
import numpy as np

def dehaze(imgs):
    opt = Config()
    model =  DehazeNet(opt.layer_num, opt.in_channels,opt.out_channels, opt.kernel_size_num, opt.kernel_size)
    model, _, _, _ = sl.load_state(opt.load_model_path, model)
    if torch.cuda.is_available():
        model = model.cuda()
        
    for img in imgs:
        hazy = Image.open(img)
        trans = Image.open('/home/ws/xh/dehaze_spyder/refined_trans/' + img.split("/")[-1]).convert("L")
        hazy = opt.transform(hazy).unsqueeze(0)
        trans = opt.transform(trans).unsqueeze(0)
        if torch.cuda.is_available():
            hazy = hazy.cuda()
            trans = trans.cuda()
        output = trans
        atmosphere1 = estimate_airlight(hazy.squeeze(0), output.squeeze(0), 'max_min')
        atmosphere2 = estimate_airlight(hazy.squeeze(0), output.squeeze(0), 'min_I')
        atmosphere3 = estimate_airlight(hazy.squeeze(0), output.squeeze(0), 'max_I')
        a_t1 = torch.mm(atmosphere1.unsqueeze(1), (1 - output.squeeze(0).view(1, -1))).view(hazy.size())
        dehazing_result1 = (hazy - a_t1) / output
        a_t2 = torch.mm(atmosphere2.unsqueeze(1), (1 - output.squeeze(0).view(1, -1))).view(hazy.size())
        dehazing_result2 = (hazy - a_t2) / output
        a_t3 = torch.mm(atmosphere3.unsqueeze(1), (1 - output.squeeze(0).view(1, -1))).view(hazy.size())
        dehazing_result3 = (hazy - a_t3) / output
        output_name1 = img.split('.')[0].split('/')[-1] + '_mm.png'
        output_name2 = img.split('.')[0].split('/')[-1] + '_mI.png'
        output_name3 = img.split('.')[0].split('/')[-1] + '_maxI.png'
        torchvision.utils.save_image(dehazing_result1, opt.test_result_mm + '/' + output_name1)
        torchvision.utils.save_image(dehazing_result2, opt.test_result_mI + '/' + output_name2)
        torchvision.utils.save_image(dehazing_result3, opt.test_result_maxI + '/' + output_name3)
               
if __name__ == '__main__':
    imgs = glob.glob('/home/ws/datasets/SOTS(Testing Set)/(indoor)nyuhaze500/hazy/*')
    dehaze(imgs)
    print("Done!")