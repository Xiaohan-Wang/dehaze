import os
from PIL import Image
from torch.utils import data
from utils.seg.watershed import watershed_seg as seg

class DehazingSet(data.Dataset):
    def __init__(self, root, transform, is_train):
        self.trans_path = root + '/trans/'
        self.gt_imgs_path = root + '/clear/'
        self.seg_path = root + '/watershed/'
        hazy_imgs = os.listdir(root + '/hazy')
        self.hazy_imgs = [root + '/hazy/' + img for img in hazy_imgs]
        self.hazy_imgs.sort()
        self.transform = transform
        self.is_train = is_train

    
    def __getitem__(self, index):
        hazy_path = self.hazy_imgs[index]
        trans_path = self.trans_path + hazy_path.split('/')[-1].split('_0.')[0] + '.png'
        gt_path = self.gt_imgs_path + hazy_path.split('_')[0].split('/')[-1] + '.png'
        trans_img = Image.open(trans_path)
        hazy_im = Image.open(hazy_path)
        gt_img = Image.open(gt_path)
        if self.is_train:
            hazy_seg = self.seg_path + hazy_path.split('/')[-1]
            seg_img = Image.open(hazy_seg).convert("L")
        else:
            seg_img = Image.fromarray(seg(hazy_im))
        if self.transform:
            trans_img = self.transform(trans_img)
            hazy_img = self.transform(hazy_im)
            seg_img = self.transform(seg_img)
            gt_img = self.transform(gt_img)
#        return hazy_img, trans_img, seg_img, gt_img, hazy_im
        return hazy_img, trans_img, seg_img, gt_img
    
    def __len__(self):
        return len(self.hazy_imgs)