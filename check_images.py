# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:09:41 2020

@author: Bill
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:04:23 2020

@author: Bill
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 05:22:29 2020


This takes a COCO annotation JSON file and displays the images and their 
segmentation and labels, just to double check that everything is ok with them, prior 
to training. 

@author: Bill and Karrington
"""
#
#  The following import gets contents of your config.py file into D. 
import data as D
  
from utils.augmentations import SSDAugmentation #, FastBaseTransform, BaseTransform

import json
import tkinter as tk
import os
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np
from PIL import ImageTk,Image
import platform as platf

class_list = []
label_map = {}
class_label = 'CLASSES:'
map_annotation =  'LABEL MAP:'
num = []
inst_dict = {}


def missing_images(js): #this function has to be in the images directory and return the name of missing image files
    file_names = []
    file_DNE = []
    
    for jsi in js['images']:
        file_names.append(jsi['file_name'])
        
    file_exist = [f for f in file_names if os.path.isfile(f)]
    file_DNE = list(set(file_exist) ^ set(file_names))
    
    print ('Missing Files: ', file_DNE)
    
def count_instances(id, js): #this function counts all of the instances
    count = 0
    for jsa in js['annotations']:
        if jsa['category_id'] == id:
            count += 1
    return(count)


def instance_collection(js): #this function makes a diction of all the instances in the json file of all the classes
    num = []
    inst_dict = {}
    for jsc in js['categories']:
        inst = count_instances(jsc['id'], js)
        num.append(inst)
        dic_name = jsc['name']
        inst_dict.update({dic_name:inst})
    return(inst_dict)

def change_cat_id(old_id, new_id, js): #this function updates the categorie id numbers
    for jsa in js['annotations']:
        if jsa['category_id'] == old_id:
            jsa['category_id'] = new_id
    for jsc in js['categories']:
        if jsc['id'] == old_id:
            jsc['id'] = new_id

def change_name(old_name, new_name, js): # this function changes the desired name of files
    for jsc in js['categories']:
        if jsc['name'] == old_name:
            jsc['name'] = new_name
        

class ImageChecker(tk.Frame):
    def __init__(self):
        # Hocus pocus...bring Tk to life...
        self.root = tk.Tk() 
        top = tk.Toplevel(self.root)
        top.withdraw()  # ...in secret....

        infile = \
            filedialog.askopenfilename(parent=top, \
                                        title='Choose JSON file')

        dataset = D.COCODetection(image_path=D.cfg.dataset.train_images,
                            info_file=infile,
                            transform=SSDAugmentation(D.MEANS))
 
        self.dataset = dataset

        try:
            self.label_map = D.KAR_LABEL_MAP
            self.classes = D.KAR_CLASSES
        except AttributeError:
            self.label_map = D.COCO_LABEL_MAP
            self.classes = D.COCO_CLASSES

        with open(infile,'r') as fp:
            print(Path(infile).stem)
            print('Loading annotations, please be patient...')
            js_all = json.load(fp)
            print('Done loading!')

        self.images_dir = '/'.join(infile.split('/')[0:-2]) + '/images/'

        self.json = js_all
        self.image_ids = js_all['images']
        self.annotations = js_all['annotations']
#        n_ann = len(js_all['annotations'])
        n_img = len(js_all['images'])
        
        self.random_image_iter = iter(np.argsort(np.random.uniform(size=(n_img))))
        self.img_display_size = (800,600)


        pad=3 # Why? 
        geom=("{0}x{1}+0+0".format(
            self.root.winfo_screenwidth()-pad, \
                self.root.winfo_screenheight()-pad))
        self.root.geometry(geom)
        
        tk.Frame.__init__(self, self.root)
        
        
        self.create_widgets()

    def create_widgets(self):   # TODO add s quit button, and gracefully exit
        self.root.bind('<Return>', self.show_next_image)
        self.grid(columnspan=2)
        
        font = 'Calibri 14'
 
        self.labtext = tk.StringVar()
        self.label = tk.Label(self.root, textvariable=self.labtext, \
                              font=font, anchor=tk.N)
 
#        self.current_comment = ''
#        self.entry = tk.Entry(self, width=80, font=font)
#        self.entry.insert(0, self.current_comment)
        
        self.canvas = tk.Canvas(self,  width=1024, height=768)
        self.show_next_image(None)
       
        self.enough = tk.Button(self, text="Enough, already!")
        self.enough.bind('<Button-1>', self.byebye)
 
        self.next = tk.Button(self, text="Show Next Image")
        self.next.bind('<Button-1>', self.show_next_image)
    
        self.label.grid(row=0,columnspan=2)
        self.canvas.grid(row=1,column=0, columnspan=2)
        self.enough.grid(row=3,column=0)
        self.next.grid(row=3,column=2)
        


#-----------------------------


    def show_next_image(self, event):
        while True:
            try:   
                i_img = next(self.random_image_iter)
                image_file = self.json['images'][i_img]['file_name']
                image_file = image_file.split('_')[-1]
#                print('image file is',image_file)
    
                try:
                    img = Image.open(self.images_dir + image_file)
                    self.img = ImageTk.PhotoImage(img.resize((self.img_display_size)))
                    width, height = self.img_display_size
                    self.canvas.create_image(width, height, \
                                          image=self.img) 
                    
                    anno = self.dataset.pull_anno(i_img)
                    # Seems crazy, but I had to subtract 1 from the label_map value...
                    for a in anno:
                        print(self.classes[self.label_map[a['category_id']]-1])
                    
                    break
                except FileNotFoundError:
                    print('oops',image_file,'does not seem to exist...')
                 
            except StopIteration:
                print('No more files! You have seen them all...')
                self.byebye(None)
                break
    

    def byebye(self, event):
        self.root.destroy()
    
    def start(self):
        self.root.mainloop()
 
    def get_slash(self): # OMG I hate computers sometimes. 
        if platf.system() == 'Windows':
            slash = '\\' 
        else:
            slash = '/'
        return slash
    

if __name__ == "__main__":
    
    ic = ImageChecker().start()
        
        
