from DehazeNet import DehazeNet
from PIL import Image
from config import Config
import glob
import torch
import torchvision

def dehaze(imgs):
    opt = Config()
    model = DehazeNet(opt.kernel_size, opt.rate_num, opt.pyramid_num, opt.conv, opt.ranking)
    model.load(opt.load_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
        
    for img in imgs:
        hazy = Image.open(img)
        hazy = opt.transform(hazy).unsqueeze(0)
        if torch.cuda.is_available():
            hazy = hazy.cuda()
        output = model(hazy)
        output_name = img.split('.')[0] + '_hf.png'
        torchvision.utils.save_image(output / 2 + 0.5, output_name)
               
if __name__ == '__main__':
    imgs = glob.glob('test_img/*')
    dehaze(imgs)
    print("Done!")