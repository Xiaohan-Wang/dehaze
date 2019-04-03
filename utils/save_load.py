# -*- coding: utf-8 -*-

"""Use this script to save and load the state of model, optimizer, epoch, and iteration."""

import time
import torch

def save_state(epoch, step, model_state, optimizer_state = None, filename = None):
    if filename == None:
        filename = time.strftime('%m%d_%H:%M:%S_epoch{}_step{}.pth'.format(epoch + 1, step ))
    model_dict = {'epoch': epoch,
                  'step': step,
                  'model_state': model_state,
                  'optimizer_state': optimizer_state
            }
    torch.save(model_dict, 'checkpoints/' + filename)
    
def load_state(load_model_path, model, optimizer = None):
    model_dict = torch.load(load_model_path)
    model.load_state_dict(model_dict['model_state'])
    if optimizer != None:
        optimizer.load_state_dict(model_dict['optimizer_state'])
    epoch, step = model_dict['epoch'], model_dict['step']
    return model, optimizer, epoch, step