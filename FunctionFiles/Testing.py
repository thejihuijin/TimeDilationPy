import numpy as np
import os
import cv2
import sys
import resize_vid
wsize_s=3.0; wsize_t=3.0; wsize_fs=5.0; wsize_f=5.0;
scaleFactor=0.125; segLength = 100.0;
dim = wsize_s*wsize_fs/scaleFactor
#%%
newrows =10; newcols=20;
print("Resizing video to [" newrows  newcols "]")
#%% Useful for video reading



