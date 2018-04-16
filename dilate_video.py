import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
import math
import dilateVideofunctions as dVf

# Oh shit nigga here we go

#%% Define Parameters
# Saliency Paremeters
wsize_s=3.0; wsize_t=3.0; wsize_fs=5.0; wsize_f=5.0;
scaleFactor=0.125; segLength = 100.0;

# Downsample rate of video when reading into memory
# Increase if running out
dim_ds = 2

# Time padding in second
# Determines how early to slow down before an interesting event
time_pad_shift = .2

# Frame rate scaling
# Scales final frame rate speed-up/slow-down by 2^fr_scale
# fr_scale = 1 yields smoother results. Currently at 1.5 to exaggerate
# effects of algorithm
fr_scale = 1.5


#%% Check Video size for saliency algorithm
# NOTE: Path must be absolute or correct relative path to video (i.e. you
# must be in the correct directory for relative path to work).
# If the video dimension does not meet the criteria of the saliency 
# algorithm, a new video will be generated
# All videos in dropbox have been tested on Mac, matlab_r2017b and show no
# issues. Some videos when being resized have caused weird distortions on
# windows. Not sure why :(
filename = ("Videos\dad_reflexes.mp4")
filename = dVf.check_video(filename, wsize_s*wsize_fs/scaleFactor)

#%% Load Video
rgbvid, fr = sliceVid(filename,0,20,dim_ds)
vid = rgbToGrayVid(rgbvid)
rows, cols, n_frames = vid.shape
del rgbvid, dim_ds

#%% Compute Saliency
print "Computing Saliency\n"
saliencyMapHolder, saliencyMapTime = compute_saliency(filename,wsize_s,wsize_t,wsize_fs,wsize_ft,scaleFactor,segLength)
print "Done\n"

saliencyMapHolder = saliencyMapHolder(:,:,0:(n_frames-1))
saliencyMapTime = saliencyMapTime(:,:,0:(n_frames-1))

#%% Compute optical flow frames
print "Computing Optical Flow\n"
flow_mags = compute_OF(vid)

#%% Compute dynamic ranges
# For display purposes only
salminmax = [min(saliencyMapHolder(:)), max(saliencyMapHolder(:))];
ofminmax = [float('Inf'),-float('Inf')]
for i in range (0,(n_frames-1))
    temp = np.min(flow_mags(:,:,i))
    if temp < ofminmax[1]
        ofminmax[1] = temp
    temp = np.max(flow_mags(:,:,i))
    if temp > ofminmax[2]
        ofminmax[2] = temp
del temp

#%% Display Saliency and Optical Flow
print "Displaying Original Video\n"
figure; colormap gray;
for i in range (0,(n_frames-1))
    st = tic
    
    imagesc(vid(:,:,i))
    title(['Original @ ' num2str(fr) ' fps - ' ...
        sprintf('%.2f',i/fr) ' seconds elapsed'])
    dur_calc = toc(st);
    pause(1/fr - dur_calc);

#%%
#
#
#                 Compare Energy Functions
#
#
    
#%% Compute Energy Functions
# Optical Flow
of_minkowski = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','MINK')
of_five_num = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','FNS')
of_weight_pool_half = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'OF','WP')

# Time Weighted Saliency
tsal_minkowski = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'TSAL','MINK')
tsal_five_num = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'TSAL','FNS')
tsal_weight_pool_half = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'TSAL','WP')

# Masked OF
mof_minkowski = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'MOF','MINK')
mof_five_num = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'MOF','FNS')
mof_weight_pool_half = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'MOF','WP')

#%% Final Graph

xvals = 1:3:n_frames
figure; hold on
if exist('gt','var')
    plot(gt,'DisplayName','Ground Truth');
end
% Plot OF
plot(xvals,of_minkowski(1:3:end),'--r*','DisplayName','OF Minkowski');
plot(xvals,of_weight_pool_half(1:3:end),'--ro','DisplayName','OF Weight Pool 1/2');
plot(xvals,of_five_num(1:3:end),'--rx','DisplayName','OF Five Num Sum');

% Plot Time Saliency
plot(xvals,tsal_minkowski(1:3:end),'--g*','DisplayName','TS Minkowski');
plot(xvals,tsal_weight_pool_half(1:3:end),'--go','DisplayName','TS Weight Pool 1/2');
plot(xvals,tsal_five_num(1:3:end),'--gx','DisplayName','TS Five Num Sum');


% Plot Masked OF
plot(xvals,mof_minkowski(1:3:end),'--b*','DisplayName','MOF Minkowski');
plot(xvals,mof_weight_pool_half(1:3:end),'--bo','DisplayName','MOF Weight Pool 1/2');
plot(xvals,mof_five_num(1:3:end),'--bx','DisplayName','MOF Five Num Sum');

legend()
title('Comparison of Various Energy Functions')
xlabel('Frame Number')
%%
clear('regex','of_*','sal_*','tsal_*','mof_*','mtsal_*');

#%%
#
#
#                 PLAYBACK VIDEOS
#
#

#%% Compute Frame Rates from energy
# select energy function
energy = compute_energy(flow_mags,saliencyMapHolder,saliencyMapTime,'MOF','WP');

# convert to fr
time_padded_fr=energy2fr(energy,fr,time_pad_shift,fr_scale)

#%% Compute playback vector from framerate vector
# resample frames based on new frame rate
adjusted_smooth_playback = fr2playback(time_padded_fr, fr);
#%% Display Energy and Framerates
figure;
subplot(121)
plot(1-energy);
xlabel('Frame Number');
ylabel('Inverted Energy');
title('Inverted Energy Graph of Video');

subplot(122);
plot(time_padded_fr);
xlabel('Frame Number');
ylabel('Frame Rate');
title('Time Dilated Frame Rate');
#%% Play Video
figure
playDilatedFrames( vid, adjusted_smooth_playback, fr, time_padded_fr )
#%% Save Video
# uncomment to save video to file

# rgbvid = vidToMat(filename);
# saveDilatedFrames( rgbvid , adjusted_smooth_playback, fr, time_padded_fr);%,outputfilename )

