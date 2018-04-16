import cv2

#%% RESIZE_VID Saves a video with new dimensions [newrows newcols]
# Input videos can be RGB or Greyscale
#
# inputName : Path to input video file
# outputName : Path and name for output video file
# newrows : Vertical size of resized video
# newcols : Horizontal size of resized video
def resize_vid(inputName,outputName,newrows,newcols):

# ECE6258: Digital image processing 
# School of Electrical and Computer Engineering 
# Georgia Instiute of Technology 
# Date Modified : 11/28/17
# By Erik Jorgensen (ejorgensen7@gatech.edu), Jihui Jin (jihui@gatech.edu)

# Read in Video
    inputReader = cv2.VideoCapture(inputName)
    
    # Prepare output video with new frame size
    outputWriter = cv2.VideoWriter(outputName,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (newrows,newcols))
    
    while(True):
        # Read in Frame
        ret, frame = inputReader.read()
        if ret == True:
            # Write new sized frame
            frame = cv2.resize(frame, (newrows, newcols))
            outputWriter.write(frame)
        else: 
            break
    inputReader.release()
    outputWriter.release()
    print('done\n')

    return