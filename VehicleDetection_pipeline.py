import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from helper_functions import *
import glob
import itertools

def extract_features(imgs, cspace='RGB', 
                        orient=9, 
                        pix_per_cell=8, 
                        cell_per_block=2,
                        hog_channel=0,
                        spatial_features =True,
                        color_features =True):
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

    

def single_img_features(image, cspace='RGB', 
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
        elif cspace == 'LAB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else: feature_image = np.copy(image)      

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        hog_image = None
        for channel in range(feature_image.shape[2]):
            features=get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True)
            
            hog_features.append(features)
        
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
        c_features = color_hist(feature_image)
        file_features.append(c_features)
    #Append spatial features if flag is set
    if spatial_features:
        s_features = bin_spatial(feature_image)
        file_features.append(s_features)
    
    if visualise:
        return np.asarray(list(itertools.chain.from_iterable(file_features))),hog_image
    else:
        return np.asarray(list(itertools.chain.from_iterable(file_features)))


def train_classifier(sample_size = 500):
    car_images = glob.glob('./vehicles/GTI_Left/*.png')
    notcar_images =glob.glob('./non-vehicles/Extras/*.png')
    cars = []
    notcars = []

    for i in range(0,sample_size):
        cars.append(car_images[i])

    for i in range(0,sample_size):
        notcars.append(notcar_images[i])

    print ('No. car images : ', len(cars))
    print ('No. not car images: ',len(notcars))

    car_ind = np.random.randint(0,len(cars))
    notcar_ind = np.random.randint(0,len(notcars))
    
    print('car img: ',car_ind,'->',cars[car_ind]) 
    print('notcar img: ',notcar_ind,'->',notcars[notcar_ind])

    car_img = cv2.imread(cars[car_ind])
    car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)

    not_car_img = cv2.imread(notcars[notcar_ind])
    not_car_img = cv2.cvtColor(not_car_img,cv2.COLOR_BGR2RGB)

    car_features1,carhog_image1 = single_img_features(car_img,
            cspace = 'YCrCb',hog_channel=0,visualise=True)
    notcar_features1,notcarhog_image1 = single_img_features(not_car_img,
            cspace='YCrCb',hog_channel= 0,visualise=True)
    

    car_features1,carhog_image1 = single_img_features(car_img,
            cspace = 'YCrCb',hog_channel=0,visualise=True)
    notcar_features1,notcarhog_image1 = single_img_features(not_car_img,
            cspace='YCrCb',hog_channel= 0,visualise=True)

    car_features2,carhog_image2 = single_img_features(car_img,
            cspace = 'YCrCb',hog_channel=1,visualise=True)
    notcar_features2,notcarhog_image2 = single_img_features(not_car_img,
            cspace='YCrCb',hog_channel= 1,visualise=True)
    
    car_features3,carhog_image3 = single_img_features(car_img,
            cspace = 'YCrCb',hog_channel=2,visualise=True)
    notcar_features3,notcarhog_image3 = single_img_features(not_car_img,
            cspace='YCrCb',hog_channel= 2,visualise=True)
    
    f1,arr1 = plt.subplots(2,2,figsize=(15,15))
    arr1[0,0].imshow(car_img)
    arr1[1,0].imshow(carhog_image2)
    arr1[0,1].imshow(carhog_image3)
    arr1[1,1].imshow(carhog_image1)

    
    f2,arr2 = plt.subplots(2,2,figsize=(15,15))
    arr2[0,0].imshow(not_car_img)
    arr2[1,0].imshow(notcarhog_image3)
    arr2[0,1].imshow(notcarhog_image2)
    arr2[1,1].imshow(notcarhog_image1)
    #print('car feature vector shape:',car_features.shape)
    #print('not car feature vector shape: ',notcar_features.shape)
    plt.show()
    
if __name__ == '__main__':
    train_classifier()
