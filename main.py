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
from Optical_Flow import compute_pyflow, compute_OF
#from Optical_Flow import compute_OF




import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Filename of video being dilated')
parser.add_argument('-ds','--downsample',help='Downsample factor', type=int)
parser.add_argument('-sal','--saliency',help='Use saliency. Defaults to no saliency',action='store_true')
parser.add_argument('-pyflow', help='Use pyflow for Optical Flow', action='store_true')
parser.add_argument('-lasso', help='Use LASSO learned weights. Must be used with saliency', action ='store_true')

args = parser.parse_args()


# read in arguments
filename = args.filename
if not args.downsample:
    dim_ds = 2
else:
    dim_ds = args.downsample

use_sam = args.saliency
use_pyflow = args.pyflow

use_lasso = args.lasso
if use_lasso:
    print('Using weights learned from lasso. Setting saliency flag to true')
    use_sam = True


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

if use_sam:
    # Saliency
    sys.path.append('sam')
    sys.path.append('sam/weights')
    from saliency import compute_SAMSal
    salmap = compute_SAMSal(rgbvid,filename)
del rgbvid

#%% Compute optical flow frames
print("Computing Optical Flow\n")
if use_pyflow:
    flow_mags = compute_pyflow(vid,filename)
else:
    flow_mags = compute_OF(vid)


# Compute Energy Functions
# Optical Flow

print("Computing Energy\n")
# of_minkowski = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','MINK');
# of_five_num = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','FNS');
# of_weight_pool_half = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','WP');

if use_lasso:
    print('Using LASSO learned weights')
    weights = np.load('weights/lasso_weights.npy')
    print('Building Features')
    features = build_features(flow_mags, salmap)
    scaledEnergy = preprocessing.scale(features)
    print('Computing Energy')
    energy = smooth_normalize(np.matmul(scaledEnergy,weights))
else:
    if use_sam:
        # compute energy using masked optical flow
        energy_map = flow_mags*salmap
    else:
        energy_map = flow_mags
    energy = minkowski(energy_map, p=2)


## Compute Frame Rates from energy
# select energy function
#energy = compute_energy(flow_mags,saliencyMapHolder(:,:,61:end),saliencyMapTime,'MOF','WP');
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



