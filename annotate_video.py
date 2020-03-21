# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:37:03 2020

@author: Bill
"""
import cv2 
#import numpy as np 


def user_pressed(key):
    wait_ms = 25
    return cv2.waitKey(wait_ms) & 0xFF == ord(key)

def key_pressed():
    return cv2.waitKey(wait_ms) & 0xFF

# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('CoVid-19.mp4') 

# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 

def you_clicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('You clicked!')

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", you_clicked)

wait_ms = 25
proceed = True
while(cap.isOpened()): 
	
    # Capture frame-by-frame 
    if proceed:
        ret, frame = cap.read() 
        
    if ret: 
        height , width , layers =  frame.shape

        new_h = height//2; new_w = width//2
        shrunk = cv2.resize(frame, (new_w, new_h)) 

    	# Display the resulting frame 
        cv2.imshow('Frame', shrunk) 
    # Break the loop 
    else: 
    	break
    
    keycode = cv2.waitKey(wait_ms) & 0xFF
    
	# Press Q on keyboard to exit 
    if keycode == ord('q'): 
        break

    if keycode==ord(' '):
        proceed = not proceed
        print(proceed)

    

# When everything done, release 
# the video capture object 
cap.release() 

# Closes all the frames 
cv2.destroyAllWindows()
