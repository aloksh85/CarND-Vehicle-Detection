import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def bin_spatial(img, size=(32,32) ):
    """
    Convert image to new color space (if specified)
    """
    temp_img = cv2.resize(img,size)
    features = temp_img.ravel() 
    # Return the feature vector
    return features

def data_look(car_list, notcar_list):
    """
    Define a function to return some characteristics of the dataset 
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = mpimg.imread(car_list[0]).shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = mpimg.imread(car_list[0]).dtype
    # Return data_dict
    return data_dict


def get_hog_features(img, orient=9, 
                          pix_per_cell=8, 
                          cell_per_block=2, 
                          vis=False, 
                          feature_vec=True):
    """
    Define a function to return HOG features and visualization
    """

    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          visualise=True, feature_vector=False,transform_sqrt=False)
                          
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          visualise=False, feature_vector=feature_vec,transform_sqrt=False)
        return features

def color_hist(img, nbins=32):
    """
    Compute the histogram of the color channels separately
    """
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def color_spatial_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        # Apply bin_spatial() to get spatial color features
        # Apply color_hist() to get color histogram features
        # Append the new feature vector to the features list
    # Return list of feature vectors
    for i in imgs:
        temp_img = mpimg.imread(i)
        if not cspace == 'RGB':
            if cspace == 'HSV':
                temp_img = cv2.cvtColor(temp_img,cv2.COLOR_RGB2HSV)
            if cspace =='LUV':
                temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2LUV)
            if color_space == 'LAB':
                temp_img= cv2.cvtColor(temp_img,cv2.COLOR_RGB2LAB)
            if color_space == 'YUV':
                temp_img = cv2.cvtColor(temp_img,cv2.COLOR_RGB2YUV)
            if color_space == 'HLS':
                temp_img = cv2.cvtColor(temp_img,cv2.COLOR_RGB2HLS)
        
        spatial_color_features = bin_spatial(temp_img)
        color_hist_features = color_hist(temp_img)
        img_features = np.concatenate((spatial_color_features, color_hist_features))
        features.append(img_features)
        
    return features

def scale_features(X):
    scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = scaler.transform(X)

    return scaled_X


def trainSVM(features,labels):
    clf =None 
    clf =LinearSVC()
    clf.fit(features,labels)
    return clf    
