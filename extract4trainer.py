# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:07:37 2020

@author: Bill
"""
from train import  NetWithLoss, CustomDataParallel, MultiBoxLoss, prepare_data

#
# I import Bolya's yolact/data subdirectory as D. This will pick
#   up information from config.py. 
#
import data as D  
from utils.augmentations import SSDAugmentation #, BaseTransform

import torch
from yolact import Yolact
import matplotlib.pyplot as plt
import matplotlib
import copy

#
# If I don't wrap this in if _name__ == '__main__', it will usually
#   fail with a broken pipe error. Something to do with multiprocessing 
#   on Windows. The phrase "Forking pickler" also appears in the 
#   traceback, and if that doesn't make you giggle, go take a break. 
#
if __name__ == '__main__':
    #
    # Define a dataset and a DataLoader, and get one datum. 
    #
    print('Before data set def, backend is',matplotlib.get_backend())
    dataset = D.COCODetection(image_path=D.cfg.dataset.train_images,
                            info_file=D.cfg.dataset.train_info,
                            transform=SSDAugmentation(D.MEANS))
    print('After data set def, backend is',matplotlib.get_backend())

#
# Defining datset somehow changes matplotlib backend, I swear to God, so I have
#   to reset it here. 
#
    matplotlib.use('Qt5Agg')    
    
    batch_size = 4
    num_workers = 1
    
    
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True, 
                                  collate_fn=D.detection_collate,
                                  pin_memory=True)
     
    for datum in data_loader:
        break
    
    
    images, (targets, masks, num_crowds) = datum
    
    #
    # Define a Yolact net, with a fancy combined prediction and loss
    #   function, and use it to process datum. 
    #
    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + D.cfg.backbone.path)
    net_plain = net
    net_plain.cuda()
    output_from_one = net_plain(datum[0][0].reshape((1,3,550,550)).cuda())
    

    criterion = MultiBoxLoss(num_classes=D.cfg.num_classes,
                             pos_threshold=D.cfg.positive_iou_threshold,
                             neg_threshold=D.cfg.negative_iou_threshold,
                             negpos_ratio=D.cfg.ohem_negpos_ratio)

    
    net_cdp = CustomDataParallel(NetWithLoss(net, criterion))
    net_cdp.cuda()
#
# Following still barfs...don't know why yet. 
#    
    losses = net_cdp(datum)


