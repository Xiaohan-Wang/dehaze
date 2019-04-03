# -*- coding: utf-8 -*-

"""Use this script to save and load the state of model, optimizer, epoch, and iteration."""

import time
import torch

def save_state(epoch, iteration, model_state, optimizer_state = None, filename = None):
    if filename == None:
        filename = time.strftime('%m%d_%H:%M:%S_epoch{}_iteration{}.pth'.format(epoch + 1, iteration + 1))
    model_dict = {'epoch': epoch,
                  'iteration': iteration,
                  'model_state': model_state,
                  'optimizer_state': optimizer_state
            }
    torch.save(model_dict, 'checkpoints/' + filename)
    
def load_state(load_model_path, model, optimizer = None):
    model_dict = torch.load(load_model_path)
    model.load_state_dict(model_dict['model_state'])
    if optimizer != None:
        optimizer.load_state_dict(model_dict['optimizer_state'])
    epoch, iteration = model_dict['epoch'], model_dict['iteration']
    return model, optimizer, epoch, iteration