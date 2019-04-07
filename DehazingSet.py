import os
from PIL import Image
from torch.utils import data

class DehazingSet(data.Dataset):
    def __init__(self, root, transform):
        self.gt_imgs_path = root + '/clear/'
        hazy_imgs = os.listdir(root + '/hazy')
        self.hazy_imgs = [root + '/hazy/' + img for img in hazy_imgs]
        self.hazy_imgs.sort()
        self.transform = transform

    
    def __getitem__(self, index):
        hazy_path = self.hazy_imgs[index]
        gt_path = self.gt_imgs_path + hazy_path.split('_')[0].split('/')[-1] + '.png' 
        gt_img = Image.open(gt_path)
        hazy_img = Image.open(hazy_path)
        if self.transform:
            gt_img = self.transform(gt_img)
            hazy_img = self.transform(hazy_img)
        return hazy_img, gt_img
    
    def __len__(self):
        return len(self.hazy_imgs)