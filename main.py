import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
import math
# import dilateVideofunctions as dVf


# Video Dilation Functions
from VideoDilation import *

# # Optical Flow
import sys
sys.path.append('./OpticalFlow')
sys.path.append('./OpticalFlow/pyflow_code/')
from Optical_Flow import compute_pyflow
#from Optical_Flow import compute_OF

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Filename of video being dilated')
parser.add_argument('-ds','--downsample',help='Downsample factor', type=int)
args = parser.parse_args()


# read in arguments
filename = args.filename
if not args.downsample:
    dim_ds = 2
else:
    dim_ds = args.downsample
print(dim_ds)


# # Actual Script

# Downsample rate of video when reading into memory
# Increase if running out
#dim_ds = 2

# Time padding in second
# Determines how early to slow down before an interesting event
time_pad_shift = .2

# Frame rate scaling
# Scales final frame rate speed-up/slow-down by 2^fr_scale
# fr_scale = 1 yields smoother results. Currently at 1.5 to exaggerate
# effects of algorithm
fr_scale = 1.5

#filename = './cat_wall_climb.mp4'


#%% Load Video
rgbvid, fr = sliceVid(filename,0,20,dim_ds)
vid = rgbToGrayVid(rgbvid)
rows, cols, n_frames = vid.shape
del rgbvid

#%% Compute optical flow frames
print("Computing Optical Flow\n")
# flow_mags = compute_OF(vid)
flow_mags = compute_pyflow(vid,filename)


# Compute Energy Functions
# Optical Flow

print("Computing Energy\n")
# of_minkowski = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','MINK');
# of_five_num = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','FNS');
# of_weight_pool_half = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','WP');
of_minkowski = minkowski(flow_mags, p=2)


## Compute Frame Rates from energy
# select energy function
#energy = compute_energy(flow_mags,saliencyMapHolder(:,:,61:end),saliencyMapTime,'MOF','WP');
energy = of_minkowski
# convert to fr
time_padded_fr=energy2fr(energy,fr,.2,1.5);



## Compute playback vector from framerate vector
# resample frames based on new frame rate
print('Resampling Frames\n')
adjusted_smooth_playback = fr2playback(time_padded_fr, fr);

vidname,ext = os.path.splitext(filename)
outname = vidname + '_dilated.avi'

print('Saving video')
saveDilatedFrames(filename,adjusted_smooth_playback.astype(int),fr,outputName=outname)



