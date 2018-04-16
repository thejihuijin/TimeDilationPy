##### opticalFlow functions #####
import numpy as np
import cv2
import os.path


#%% COMPUTE_OF returns the optical flow magnitudes for a given video
# Currently uses Horn Schunck
# Assumes input video is in grey scale
# Dimensions = (rows, cols, frames)
#
# vid : 3D video matrix
#
# flow_mags : 3D Optical Flow magnitudes

def compute_OF(vid)

# ECE6258: Digital image processing 
# School of Electrical and Computer Engineering 
# Georgia Instiute of Technology 
# Date Modified : 11/28/17
# By Erik Jorgensen (ejorgensen7@gatech.edu), Jihui Jin (jihui@gatech.edu)

#    [rows,cols,n_frames] = size(vid)
#    OF = opticalFlowHS()
#    flow_mags = zeros(rows,cols,n_frames)
#    for i = 1:n_frames
#        # Store magnitude frames
#        flow = estimateFlow(OF, vid(:,:,i))
#        flow_mags(:,:,i) = flow.Magnitude
#    return flow_mags

vid = cv2.VideoCapture(vid)        
ret, frame1 = vid.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = vid.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    prvs = next
cap.release()
cv2.destroyAllWindows()


