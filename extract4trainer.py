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

from utils.augmentations import SSDAugmentation #, FastBaseTransform, BaseTransform
import torch
from yolact import Yolact
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import copy

def npscl(xin):
    x = copy.copy(xin)
    for i in range(x.shape[2]):
        xx = x[:,:,i]
        xmax = np.max(xx); xmin = np.min(xx)
        if xmax > xmin:
            x[:,:,i] = (xx-xmin)/(xmax-xmin)
        else:
            print('All the same!!',i)

    return x

def myshow(img):
    if img.size()[0] != 3:
        maskshow(img)
        
    ishow = img.cpu().numpy().transpose((1,2,0))
    ishow = (ishow-np.min(ishow))/(np.max(ishow) - np.min(ishow)).astype(np.int)
    plt.imshow(ishow)

def maskshow(img,pick=None):
    isize = img.size()
    if not pick:
        nmask = isize[0]
        pick = list(range(nmask))
    else:
        if type(pick) is not list:
            pick = list(pick)

    mask = np.zeros(isize[1:])
    img = img.cpu().detach().numpy()
    for i in pick:
        mask += img[i,:,:]
    
    plt.imshow(mask)
    rtn_size = list(mask.shape)
    rtn_size.append(1)
    return mask.reshape(rtn_size);

#import copy

#
# If I don't wrap this in if _name__ == '__main__', it will usually
#   fail with a broken pipe error. Something to do with multiprocessing 
#   on Windows. The phrase "Forking pickler" also appears in the 
#   traceback, and if that doesn't make you giggle, go take a break. 
#
if __name__ == '__main__':
    backend_I_want = 'Qt5Agg'
    #
    # Define a dataset and a DataLoader, and get one datum. 
    #
    print('Before data set def, backend is',matplotlib.get_backend())
    dataset = D.COCODetection(image_path=D.cfg.dataset.train_images,
                            info_file=D.cfg.dataset.train_info,
                            transform=SSDAugmentation(D.MEANS))
    print('After data set def, backend is',matplotlib.get_backend())

#
# Defining datset somehow sometiems changes matplotlib backend to Agg, I 
#   swear to God, so I reset it here if ncecessary.  
#
    if matplotlib.get_backend() != backend_I_want:
        print('WTF? Resetting matplotlib backend...')
        matplotlib.use(backend_I_want)    
    
    batch_size = 4
    num_workers = 0
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True, 
                                  collate_fn=D.detection_collate,
                                  pin_memory=True)
    
    
    data_loader_iterator = iter(data_loader)
    
    datum = next(data_loader_iterator)    
    
    #  datum itself is a list of 2 lists. The first has length batch_size and 
    #   contains images, the second has length 3 and contains targets, masks, 
    #   and num_crowds. 
    #
    #  images are 3x550x550 tensors, in a list of length batch_size.
    #
    #  targets are nx5 tensors, what is n?  in a list of length batch_size.
    #
    #  masks are tensors, size kx550x550, list of length batch_size, where k is
    #   the number of objects in the corresponding image. 
    #
    #  crowds are numpy.int32, in a list of length batch_size.
    #  
    
    images, (targets, masks, num_crowds) = datum
    #                 img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
#                    {'num_crowds': num_crowds, 'labels': target[:, 4]})

    #
    # Define a Yolact net, with a fancy combined prediction and loss
    #   function, and use it to process datum. 
    #
    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + D.cfg.backbone.path)

    criterion = MultiBoxLoss(num_classes=D.cfg.num_classes,
                             pos_threshold=D.cfg.positive_iou_threshold,
                             neg_threshold=D.cfg.negative_iou_threshold,
                             negpos_ratio=D.cfg.ohem_negpos_ratio)

    
    net_cdp = CustomDataParallel(NetWithLoss(net, criterion))
    net_cdp.cuda()

    losses = net_cdp(datum)
    losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
    loss = sum([losses[k] for k in losses])
    
    print('Whoa! We made it! Loss is',loss)

#
#  Below here I am trying to call the augmentation by hand, to quickly test
#    my work using a single datum.     

#    tform=SSDAugmentation(D.MEANS)
#    
#    img = np.array(images[0].cpu()).transpose((1,2,0))
#    img0 = copy.copy(img)
#    mask = np.array(masks[0].cpu())
#    target = targets[0]
#    target = np.array(target.cpu())
#    nc = num_crowds[0]
#    img, mask, boxes, labels=\
#    tform(img, mask, target[:, :4],
#                    {'num_crowds': nc, 'labels': target[:, 4]})
#
#    plt.imshow(npscl(img0))
#
#    plt.imshow(npscl(img))
#
#



