from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
#import argparse
import sys
sys.path.append('./pyflow_code/')
import pyflow
import os
import cv2


def compute_pyflow(vid,vidpath):
    # Check vid path to load video
    pathname, ext = os.path.splitext(vidpath)
    flowpath = pathname+ '_flow.npy'
    _, name = os.path.split(flowpath)
    print('Checking for', name) 
    if os.path.exists(flowpath):
        flow_mags = np.load(flowpath)
        if flow_mags.shape == vid.shape:
            print('file found')
            return np.load(flowpath)
        print('Flow mag shape does not match')
    else:
        print(name, 'does not exist. Computing from scratch.')
   


    # calculate flow from scratch 
    # vid must be double from 0 to 1
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    rows, cols, n_frames = vid.shape
    flow_mags = np.zeros(vid.shape)
    
    update = int(n_frames/10)
    frame1 = vid[...,0,np.newaxis].copy(order='C')
    for i in range(1,n_frames):
        frame2 = vid[...,i,np.newaxis].copy(order='C')
        u, v, im2W = pyflow.coarse2fine_flow(
                    frame1, frame2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                    nSORIterations, colType)
        mag = (u**2 + v**2)**.5       
        flow_mags[...,i] = cv2.normalize(mag,None,0,1.0,cv2.NORM_MINMAX)
        
        frame1 = frame2.copy()
        if i % update == 0:
            print(int(100*i/n_frame),'% complete')
    flow_mags[...,0] = flow_mags[...,1]
    np.save(flowpath,flow_mags)
    return flow_mags
