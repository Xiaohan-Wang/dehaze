from DehazeNet import DehazeNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
import Config

def test(opt):
    model = DehazeNet(opt.kernel_size, opt.rate_num,opt.conv, opt.ranking)
    model.load(opt.load_model_path)
    
    test_set = DehazingSet(opt.test_data_root)
    dataloader = DataLoader(test_set,opt.batch_size, shuffle = True, num_workers = opt.num_workers)
    
    loss_meter = meter.AverageValueMeter()
    
    for iteration, (hazy_img, gt_img) in enumerate(dataloader):
        input_data = Variable(hazy_img)
        target_data = Variable(gt_img)
        
        output_result = model(input_data)
        loss = nn.MSELoss()(output_result, target_data)
        
        loss_meter.add(loss.data[0])
    
    print("Loss at Test set:", loss_meter.value()[0])      

if __name__ == '__main__':
    opt = Config()
    test(opt)