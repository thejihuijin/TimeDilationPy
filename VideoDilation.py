# # Video Dilation Functions
import numpy as np
import cv2
from skimage.transform import rescale, resize
import os.path
import math
import pandas as pd

###############################################
#
#               Video IO
#
###############################################

def sliceVid( filename, startTime, endTime, ds ):
# ECE6258: Digital image processing 
# School of Electrical and Computer Engineering 
# Georgia Instiute of Technology 
# Date Modified : 11/28/17
# By Erik Jorgensen (ejorgensen7@gatech.edu), Jihui Jin (jihui@gatech.edu)

    vid = cv2.VideoCapture(filename)    
    rows = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cols = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    fr = math.floor(vid.get(cv2.CAP_PROP_FPS))
    
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #Convert start and end times to frame numbers
    start_frame = math.floor(startTime*fr)
    end_frame = math.floor(endTime*fr)
    if end_frame > total_frames:
        end_frame = total_frames
    
    num_frames = int(end_frame - start_frame)
    
    print('Reading in', num_frames,'frames')
    # Downsampled vid size
    rows_ds = math.floor(rows/ds)
    cols_ds = math.floor(cols/ds)
    
    vidMatrix = np.zeros((rows_ds, cols_ds, 3, num_frames))

    # throw away initial frames
    for i in range(start_frame):
        ret, frame = vid.read()
        if not ret:
            return None
        
    # grab frames
    for i in range(num_frames):
        ret, frame = vid.read()
        if ret:
            frame = resize(frame, (rows_ds, cols_ds, 3)).astype(np.float32)
            vidMatrix[...,i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i % 50 == 0:
                print(str(i)+'/'+str(num_frames))
        else:
            break
        
    return vidMatrix, fr

def rgbToGrayVid( rgbVidMatrix ):
# ECE6258: Digital image processing 
# School of Electrical and Computer Engineering 
# Georgia Instiute of Technology 
# Date Modified : 11/28/17
# By Erik Jorgensen (ejorgensen7@gatech.edu), Jihui Jin (jihui@gatech.edu)

# Sequentially convert each RGB frame to grayscale
    rows, cols, _, frames = rgbVidMatrix.shape
    print('Converting', frames, 'frames to grayscale')
    grayVidMatrix = np.zeros((rows,cols,frames))
    for i in range (frames):
        grayVidMatrix[...,i] = cv2.cvtColor(rgbVidMatrix[...,i].astype(np.float32)
                                            ,cv2.COLOR_RGB2GRAY)
    
    return(grayVidMatrix)

def saveDilatedFrames(filename, frameIndices, fr, outputName='test.avi'):
    print(filename)
    cap = cv2.VideoCapture(filename)
    
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #fr = math.floor(vid.get(cv2.CAP_PROP_FPS))
    print(H,W)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #H,W = vidMat.shape[:2]
    out = cv2.VideoWriter(outputName,fourcc,fr,(W,H))
    
    findex = 0
    adjfindex = 0
    ret,frame = cap.read()
    while(ret):
        while findex == frameIndices[adjfindex]:
            out.write(frame)
            adjfindex += 1
            if adjfindex >= len(frameIndices):
                out.release()
                cap.release()
                return
        findex += 1
        ret, frame = cap.read()

    out.release()
    cap.release()   
###############################################
#
#              Energy Map Manipulations 
#
###############################################
def logsal(img):
    # handles log 0
    img[img==0] = np.min(img[np.nonzero(img)])/2
    return np.log10(img)


def normalize(img):
    img[np.isinf(img)] = np.nan
    img -= np.nanmin(img)
    img[np.isnan(img)] = 0.0
    img /= np.max(img)
    return img

def otsu_threshold(img):
    # Assumes input image is [0,1] float
    _, th = cv2.threshold((255*img).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (th/255).astype('bool')


###############################################
#
#              Pooling Functions 
#
###############################################
def weight_pool(frames, p):
    n_frames = frames.shape[2]
    energy = np.zeros((n_frames,))
    for i in np.arange(n_frames):
        tmpframe = frames[...,i][frames[...,i] != 0]  
        if tmpframe.any():
            energy[i] = (tmpframe**(1+p)).sum() / (tmpframe**p).sum()
    return smooth_normalize(energy)

def total_energy(frames):
    n_frames = frames.shape[2]
    energy = np.sum(frames,axis=(0,1))
    return smooth_normalize(energy)

def minkowski(frames, p):
    n_frames = frames.shape[2]
    frames = frames**p
    energy = np.zeros((n_frames,))
    for i in np.arange(n_frames):
        tmp = frames[...,i][frames[...,i] != 0]
        if tmp.any():
            energy[i] = tmp.mean()
    return smooth_normalize(energy)

def five_num_sum(frames):
    n_frames = frames.shape[2]
    energy = np.zeros((n_frames,))
    for i in np.arange(n_frames):
        tmpframe = frames[...,i][frames[...,i] != 0]
        if tmpframe.any():
            energy[i] = tmpframe.mean() + tmpframe.max() + np.percentile(tmpframe,[25,50,75]).sum()
            energy[i] /= 5
    return smooth_normalize(energy)

###############################################
#
#          Feature Learning/Building 
#
###############################################
def build_features(flow_mag, saliency):
    # Assumes video shapes = (frames,height,width) = (N,H,W)
    # Assumes Flow magnitude video has one less frame than saliency
    # Returns matrix that is (N-1,40)
    
    # Compute saliency mask with Otsu's thresholding per frame, Count number of connected components per frame
    (H,W,N) = saliency.shape
    sals_mask = np.zeros(saliency.shape)
    num_comps = np.zeros((N,))
    for i in np.arange(N):
        sals_mask[...,i] = otsu_threshold(saliency[...,i])
        _, num_comps[i] = ndimage.label(sals_mask[...,i])
    masked_flow_mag = np.multiply(flow_mag, sals_mask)
    # masked_flow_mag = np.multiply(flow_mag[...,1:], sals_mask[...,1:])
    nnz_sal_mask = np.count_nonzero(sals_mask, axis=(0,1))
    
    # Compute original feature vectors
    flowFeatures = six_features(flow_mag)                       # OF Features
    salFeatures = six_features(saliency)                            # Saliency features
    salMaskFeatures = np.column_stack((nnz_sal_mask,num_comps)) # Saliency mask features
    salMaskedOFFeatures = six_features(masked_flow_mag)         # Saliency-masked Optical Flow features
    
    # Build data matrix with raw energy vectors
    energyMatOrig = np.column_stack((flowFeatures, salFeatures, salMaskFeatures, salMaskedOFFeatures))

    # Build data matrix with smoothed energy vectors
    energyMatSmooth = np.zeros(energyMatOrig.shape)
    for i in np.arange(energyMatOrig.shape[1]):
        energyMatSmooth[:,i] = smooth_normalize(energyMatOrig[:,i], 15, 4)

    # Combine raw and smoothed energy matrices into one final matrix
    energyMatBoth = np.column_stack((energyMatOrig, energyMatSmooth))
    
    feature_labels = ['Average OF','Median OF', 'Weight Pooled OF', 'Total Energy OF', 'Minkowski OF','5 Num Sum OF', \
                  'Average Saliency', 'Median Saliency', 'Weight Pooled Saliency', 'Total Energy Saliency', \
                  'Minkowski Saliency', '5 Num Sum Saliency', 'Num Nonzeros Saliency Mask',\
                  'Num Connected Components Saliency Mask', 'Average Sal-Masked OF', 'Median Sal-Masked OF', \
                  'Weight Pooled Sal-Masked OF', 'Total Energy Sal-Masked OF', 'Minkowski Sal-Masked OF',\
                  '5 Num Sum Sal-Masked OF', 'Smoothed Average OF', 'Smoothed Median OF', 'Smoothed Weight Pooled OF', \
                  'Smoothed Total Energy OF', 'Smoothed Minkowski OF', 'Smoothed 5 Num Sum OF', 'Smoothed Average Saliency', \
                  'Smoothed Median Saliency', 'Smoothed Weight Pooled Saliency', 'Smoothed Total Energy Saliency', \
                  'Smoothed Minkowski Saliency', 'Smoothed 5 Num Sum Saliency', 'Smoothed Num Nonzeros Saliency Mask',\
                  'Smoothed Num Connected Components Saliency Mask', 'Smoothed Average Sal-Masked OF', \
                  'Smoothed Median Sal-Masked OF', 'Smoothed Weight Pooled Sal-Masked OF', \
                  'Smoothed Total Energy Sal-Masked OF', 'Smoothed Minkowski Sal-Masked OF','Smoothed 5 Num Sum Sal-Masked OF']
    return pd.DataFrame(data=energyMatBoth,columns=feature_labels)


def learn_weights(feature_mat, ground_truth, cutoff_accuracy):
    # Assumes feature_mat an ground_truth are already pre-scaled, smoothed, and zero-mean
    # cutoff_accuracy is minimum allowed R-squared value [0,1] of fit to ground truth
    trials = 1000
    alphas = np.logspace(-5,1,trials)
    for i,a in enumerate(alphas):
        lasso = linear_model.Lasso(alpha=a, max_iter=10000)
        lasso.fit(feature_mat, ground_truth)
        
        # If we went below cutoff accuracy, return the previous fitting that was still good enough
        score = lasso.score(feature_mat, ground_truth)
        if score < cutoff_accuracy:
            if i == 0:
                alph = alphas[i]
            else:
                alph = alphas[i-1]
            lasso = linear_model.Lasso(alpha=alph, max_iter=10000)
            lasso.fit(feature_mat, ground_truth)
            score = lasso.score(feature_mat, ground_truth)
            return lasso.coef_, alph, score
    return lasso.coef_, alphas[i], score
###############################################
#
#         Energy Vector Manipulations 
#
###############################################

def smooth_normalize(energy, mov_avg_window=15, mov_med_window=5):
    # Smooth with running median and average
    smoothed = pd.rolling_median(energy, mov_med_window, min_periods=1, center=True)
    smoothed = pd.rolling_mean(smoothed, mov_avg_window, min_periods=1, center=True)
    
    # Normalize to [0,1]
    smoothed = smoothed - smoothed.min()
    smoothed = smoothed / smoothed.max()
    return smoothed

def energy2fr(energy,fr,time_pad=.2,scale=1):
    energy_normal = 1-energy;

    # Scale framerate to speed up
    # Adjust framerate to exponential
    fr_scaled = fr*2**(scale*(2*(energy_normal-np.mean(energy_normal))));


    # time pad frame rate
    fr_adj = adjustFR( fr_scaled, time_pad, fr );

    # Smooth the adjusted framerate
    mov_avg_window = 5;
    fr_adj_smooth = pd.rolling_mean( fr_adj, mov_avg_window ,min_periods=1);
    fr_adj_smooth = pd.rolling_mean( fr_adj_smooth, mov_avg_window,min_periods=1 );

    return fr_adj_smooth
    
def adjustFR(frVect, timeshift, fr):
    slope = np.convolve(frVect, [1, -1], mode='valid')
    shift = int(timeshift*fr)
    slope_adj = np.array(slope)
    
    for i in np.arange(shift,len(slope)-shift).astype('int'):
        if slope[i] < 0:
            slope_adj[i-shift] += slope[i]
        elif slope[i] > 0:
            slope_adj[i+shift] += slope[i]
        slope_adj[i] -= slope[i]
    
    frAdjusted = np.cumsum(np.insert(slope_adj, 0, 0)) + frVect[0]
    frAdjusted = np.minimum(frAdjusted, frVect.max())
    frAdjusted = np.maximum(frAdjusted, frVect.min())
    return frAdjusted


def fr2playback( frameRates, playback_fr ):
    #Delays between consecutive frames at variable framerate
    dilated_delays = 1/frameRates

    # Cumulative time until each frame at variable framerate
    dilated_times = np.append(np.zeros(1), np.cumsum(dilated_delays))
    #[0 cumsum(dilated_delays(1:end))];

    # Cumulative time until each frame at constant framerate
    plybk_times = np.arange(0,dilated_times[-1],1.0/playback_fr)
    #0 : 1/playback_fr : dilated_times(end);

    # Find each frame from the variable framerate delays vector that occurs
    # closest to each frame in the constant framerate delay vector.
    playbackFrames = np.zeros(plybk_times.shape)
    #zeros(1,length(plybk_times));
    frame_ptr = 0;
    for i,plybk_time in enumerate(plybk_times):
        min_dist = np.inf
        while 1:
            if frame_ptr + 1 >= len(dilated_times):
                if frame_ptr >= len(playbackFrames):
                    frame_ptr = len(playbackFrames)-1
                playbackFrames[i] = frame_ptr-1;
                break

            curr_dist = np.abs( dilated_times[frame_ptr] - plybk_times[i] );
            next_dist = np.abs( dilated_times[frame_ptr + 1] - plybk_times[i] );

            if curr_dist < min_dist:
                min_dist = curr_dist

            if next_dist > curr_dist:
                playbackFrames[i] = frame_ptr
                break

            frame_ptr += 1
    return playbackFrames

