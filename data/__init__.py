from .config import *
from .coco import *
# Ok, next line is just me testing to see how this works
#from .config3March2020meeting import *
#
#  I guess what it does is that "import .config" will import everything in 
#    config.py.  Supposedly the dot indicates current directory, which I guess 
#    must be yolact/data , since that is where this script resides and removing
#    the dot causes python to not find the module. 

import torch
import cv2
import numpy as np
