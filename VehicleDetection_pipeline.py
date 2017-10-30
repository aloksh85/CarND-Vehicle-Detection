import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from helper_functions import *



def extract_features(imgs, cspace='RGB', 
                        orient=9, 
                        pix_per_cell=8, 
                        cell_per_block=2,
                        hog_channel=0,
                        spatial_features =True,
                        color_features =True)
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        file_features =[]
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
	#Append color histogram features if flag is set
        if color_features:
            c_features = color_hist(feature_image,cspace)
            file_features.append(s_features)
        #Append spatial features if flag is set
        if spatial_features:
            s_features = bin_spatial(feature_image)
            file_features.append(s_features)

        features.append(np.concatenate(file_features))

    # Return list of feature vectors
    return features

    

def single_img_features(img, cspace='RGB', 
                        orient=9, 
                        pix_per_cell=8, 
                        cell_per_block=2,
                        hog_channel=0,
                        visualise=False,
                        spatial_features=True,
                        color_features=True):

    file_features =[]
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        if visualise:
            hog_features,hog_image = get_hog_features(feature_image[:,:,hog_channel], 
                    orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        else:

            hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                    orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)

    # Append the new feature vector to the features list
    file_features.append(hog_features)
    #Append color histogram features if flag is set
    if color_features:
        c_features = color_hist(feature_image,cspace)
        file_features.append(s_features)
    #Append spatial features if flag is set
    if spatial_features:
        s_features = bin_spatial(feature_image)
        file_features.append(s_features)

    if visualise:
        return np.concatenate(file_features),hog_image
    else:
        return np.concatenate(file_features)


def train_classifier():
   image = cv2.imread() 



