# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 05:22:29 2020

This shows how to combine two COCO annotation JSON files. Since these 
annotation files are dictionaries, each item of which contains a list, I can 
iterate over the dict keys, and append the elements of one list to the other
list. 

This is NOT yet a script you can just run. On Windows, I found I could not get 
the uichoosefile function to work. It may work for you on Ubuntu; F9 the def and
test it. Do the imports too, of course. 

If uichoosefile works from the console prompt, go ahead and uncomment the 
calls to it below, delete the hard-coded file names,delte wd, and use the 
script to combine COCO annotation files. If the files are big, it might take
several minutes. 

If uichoosefile does NOT work at the console prompt, then you will need to 
edit the file names below (js1 and js2), and also the working directory (wd) so
that they make sense on your machine. 

Just to summarize: js1, js2, and wd are hard-coded for my machine in this script.
You can't use them as they are. You can either use uichoosefile, if it works, to
get sensible values for them, or you can hard-code your own sensible values. 

@author: Bill
"""

import json
import tkinter as tk
from tkinter.filedialog import FileDialog
import os
import time
import pathlib
from pathlib import Path

class_list = []
label_map = {}
class_label = 'CLASSES:'
map_annotation =  'LABEL MAP:'


def uichoosefile(title = None, initialdir = None):
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilename(title=title, initialdir = initialdir)
    return filename

def date_for_filename():
    tgt = time.localtime()
    year = str(tgt.tm_year)
    mon = "{:02}".format(tgt.tm_mon)
    day = "{:02}".format(tgt.tm_mday)
    hour = "{:02}".format(tgt.tm_hour)
    minute = "{:02}".format(tgt.tm_min)
    datestr = '~' + year + '-' + mon + '-' + day + '_' + hour + minute
    return datestr

if __name__ == "__main__":
    js1 = uichoosefile(title='Choose first JSON file...')
    with open(js1,'r') as fp:
        file1 = json.load(fp)

    js2 = uichoosefile(title='Choose second JSON file...')
    with open(js2,'r') as fp:
        file2 = json.load(fp)
        
 
    # js1_id_int = file1['annotations'][-1]['id']
    # js1_imageid_int = file1['annotations'][-1]['image_id']
    # js1_catid_int =file1['annotations']
    
    js1_img_int = file1['images'][-1]['id'] + 1
    js1_cat_int = file1['categories'][-1]['id'] +1
    js1_ann_id_int = file1['annotations'][-1]['id'] +1
    
    js2_img = file2['images']
    js2_cat = file2['categories']
    js2_ann = file2['annotations']
  
    
    for i in range (len(js2_img)):
        js2_img[i]['id'] = js1_img_int +  js2_img[i]['id'] 
        
    for i in range(len(js2_cat)):
        js2_cat[i]['id'] = js2_cat[i]['id'] + js1_cat_int 
        
    for i in range(len(js2_ann)):
        js2_ann[i]['id'] = js1_ann_id_int + js2_ann[i]['id'] 
        js2_ann[i]['image_id'] = js1_img_int + js2_ann[i]['image_id'] 
        js2_ann[i]['category_id'] = js1_cat_int + js2_ann[i]['category_id'] 
        
    
    req_keys = ('images','annotations','categories') # other keys ignored. 
    for key in req_keys:
        for element in file2[key]:
            file1[key].append(element)
    wd = os.getcwd() + '/'
    js3 = wd + Path(js1).stem + Path(js2).stem + date_for_filename()+'.json'
    # tread = open(js3 , 'w')
    # tread.write(str(file1))
    # tread.close()
    with open(js3,'w') as fp:
        json.dump(file1,fp)
        
    with open(js3,'r') as fp:
        file3 = json.load(fp)
        
    js3_cat = file3['categories']
    
    for i in range(len(js3_cat)):
        class_list.append(js3_cat[i]['name'])
        label_map.update({js3_cat[i]['id']:i+1})
        
    txt = wd + Path(js1).stem + Path(js2).stem + date_for_filename()+ '.txt'

    f = open(txt, 'w')
    f.write('LABEL MAP:\n')
    f.write(str(label_map))
    f.write('\n')
    f.write('CLASSES\n')
    f.write(str(class_list))
    f.close()
    
    
    
