
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class DehazingSet(data.Dataset):
    def __init__(self, root):
        gt_imgs = os.listdir(root + '/gt')
        hazy_imgs = os.listdir(root + '/hazy')
        self.gt_imgs = [root + '/gt/' + img for img in gt_imgs]
        self.hazy_imgs = [root + '/hazy/' + img for img in hazy_imgs]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5]),
            T.CenterCrop(32)
        ])
    
    def __getitem__(self, index):
        gt_path = self.gt_imgs[index]
        hazy_path = self.hazy_imgs[index]
        gt_img = Image.open(gt_path)
        if self.transform:
            gt_img = self.transform(gt_img)
            hazy_img = self.transform(hazy_img)
            hazy_img = Image.open(hazy_path)
        print("gt Image {}: {}".format(index, gt_img.size()))
        print("hazy Image {}: {}".format(index, hazy_img.size()))
        return hazy_img, gt_img
    
    def __len__(self):
        return len(self.gt_imgs)