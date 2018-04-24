# VideoDilation

This repo contains preliminary code for video dilation.
Video Dilation is an automated frame rate adjustment of videos 
depending on the energy present in each frame. More "interesting" events
are presented in slow motion while as less "interesting" events are
sped through.

## Authors
Jihui Jin, Erik Jorgensen, Christian De La Pena

## Contents
- main.py: Contains main script
- VideoDilation.py: Contains video dilation functions
- sam: Contains Saliency code using SAM-VGG
- Optical Flow: Contains OF code for pyflow and cv2's farneback optical flow estimation

## Dependencies
- Videos: Recommend short clips ~30 seconds long or less with minimal camera movement.
- Ran with Python 3.6
- Libraries:
  - numpy
  - matplotlib
  - OpenCV 3
  - ffmpeg
  - Scikit Image
  - Pandas
  - Scikit Learn
  - PIL
- For Pyflow based Optical Flow:
  - cython ([useful link for windows](https://github.com/ContinuumIO/anaconda-issues/issues/2449))
- For SAM-VGG Saliency:
  - Keras
  - Theano (Please see saliency directory readme for exact version)
  
Example videos can be found [here](https://www.dropbox.com/sh/wpze1o1taqz6yyh/AABEjfNdWdxFotm40nC9Dp_ma?dl=0 "Test Videos").
Original Paper can be found [here](https://www.dropbox.com/s/m8o7xxb9lnl4zrz/JinJihui_JorgensenErik.pdf?dl=0).
  
## How To Run
An example script is provided in `main.py`
To run the basic video dilation pipeline using only farneback optical flow for energy maps and minkowski for energy pooling:
```
python main.py pathtovidfile
```

To use saliency, add the `-sal` flag. Please ensure your computer is configured to use Theano and Keras. Note that without a GPU this may take awhile to generate the energy maps.
```
python main.py pathtovidfile -sal
```
To use pyflow, add the `-pyflow` flag. Note that pyflow computations are expensive and can take awhile
```
python main.py pathtovidfile -pyflow
```
To use the weights learned via LASSO on `dad_reflexes.mp4`, add the flag `-lasso`. Note that this turns on `-sal` by default
```
python main.py pathtovidfile -pyflow -lasso
```
To reduce computational load or memory usage, specify a downsampling factor for energy maps. The default is 2
```
python main.py pathtovidfile -ds 4
```




