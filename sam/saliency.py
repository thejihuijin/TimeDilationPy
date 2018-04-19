from __future__ import division
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import os, cv2, sys
import numpy as np
from config import *
from utilities import preprocess_imagemat, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss

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


def generator_from_mat(b_s, vid):
    vid = vid.transpose((3,0,1,2))
    counter = 0
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))
    while True:
        yield [preprocess_imagemat(vid[counter:counter+b_s],shape_r,shape_c), gaussian]
        counter = (counter + b_s) % len(vid)



def compute_SAMSal(rgbvid, filename):
    # Check if currently exists
    outname,ext = os.path.splitext(filename)
    outname += '_sam.npy'
    print('Checking for', outname)
    if os.path.exists(outname):
        salmap = np.load(outname)
        if salmap.shape[-1] == rgbvid.shape[-1]:
            print('file found')
            return salmap
        print('Saliency Map length incorrect. Recomputing')
    else:
        print(outname, 'does not exist. Computing from scratch')
    
    
    x = Input((3, shape_r, shape_c))
    x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

    m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
    print("Compiling SAM-VGG")
    m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])


    nb_imgs_test = rgbvid.shape[-1]

    if nb_imgs_test % b_s != 0:
        print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        exit()

    print("Loading SAM-VGG weights")
    m.load_weights('./sam/weights/sam-vgg_salicon_weights.pkl')

    print("Predicting saliency maps")
    predictions = m.predict_generator(generator_from_mat(b_s=b_s, vid=rgbvid), nb_imgs_test)[0]

    H,W,_,n = rgbvid.shape
    output = np.zeros((H,W,n))
    for i,pred in enumerate(predictions):
        res = postprocess_predictions(pred[0], H, W)
        output[...,i] = normalize(logsal(res))

        #name = 'frame' + str(i) + '.png'
        #original_image = cv2.imread(imgs_test_path + name, 0)
        #res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        #cv2.imwrite(output_folder + '%s' % name, res.astype(int))

    outname,ext = os.path.splitext(filename)
    outname += '_sam.npy'
    np.save(outname,output)
    return output
