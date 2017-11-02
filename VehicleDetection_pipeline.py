import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from helper_functions import *
import glob
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    print('extracting features for ', len(imgs), 'imgs')

    for img_file in imgs:
        # Read in each one by one
        image = cv2.imread(img_file)
        imgae = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            c_features = color_hist(feature_image)
            file_features.append(c_features)
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

def visualise_feature_image(car_img, notcar_img,cspace='YCrCb'):
    
    car_features1,carhog_image1 = single_img_features(car_img,
            cspace = cspace,hog_channel=0,visualise=True)
    notcar_features1,notcarhog_image1 = single_img_features(not_car_img,
            cspace=cspace,hog_channel= 0,visualise=True)

    car_features1,carhog_image1 = single_img_features(car_img,
            cspace = cspace,hog_channel=0,visualise=True)
    notcar_features1,notcarhog_image1 = single_img_features(not_car_img,
            cspace=cspace,hog_channel= 0,visualise=True)

    car_features2,carhog_image2 = single_img_features(car_img,
            cspace = cspace,hog_channel=1,visualise=True)
    notcar_features2,notcarhog_image2 = single_img_features(not_car_img,
            cspace=cspace,hog_channel= 1,visualise=True)
    
    car_features3,carhog_image3 = single_img_features(car_img,
            cspace = cspace,hog_channel=2,visualise=True)
    notcar_features3,notcarhog_image3 = single_img_features(not_car_img,
            cspace=cspace,hog_channel= 2,visualise=True)
    
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
    plt.show()

def train_classifier(train_features,train_labels,clf_type='SVM'):

    if clf_type =='SVM':
        print ('Training SVM')
        clf = train_SVM(train_features,train_labels)
        print('Training complete.')

    return clf

    # split traiin test
    # train SVM
    # check accuracy
    # sliding window search
    # smoothing between frames
    #Train YOLO



def vehicle_detection_training(test_caassifier =False):
    car_images = glob.glob('./vehicles/*.png')
    notcar_images =glob.glob('./non-vehicles/*.png')
    cars = []
    notcars = []

    sample_size = 1000

    for img in car_images:
        cars.append(img)

    for img in notcar_images:
        notcars.append(img)
   
    print ('No. car images : ', len(cars)) 
    print ('No. not car images: ',len(notcars))

    #car_ind = np.random.randint(0,len(cars),sample_size)
    #notcar_ind = np.random.randint(0,len(notcars),sample_size)
   
    v_car_ind = np.random.randint(0,len(cars))
    v_notcar_ind = np.random.randint(0,len(notcars))
    print('car img: ',v_car_ind,'->',cars[v_car_ind]) 
    print('notcar img: ',v_notcar_ind,'->',notcars[v_notcar_ind])

    car_img = cv2.imread(cars[v_car_ind])
    car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)

    not_car_img = cv2.imread(notcars[v_notcar_ind])
    not_car_img = cv2.cvtColor(not_car_img,cv2.COLOR_BGR2RGB)
    
    #visualise_feature_image(car_img,notcar_img,cspace='YCrCb')
    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)
    
    train_labels = np.hstack((np.ones(len(cars)),np.zeros(len(notcars))))
    train_features = np.vstack((car_features,notcar_features))
    scaled_train_features = scale_features(train_features)

    print('scaled features shape: ',scaled_train_features.shape)
    print('labels vector shape: ', train_labels.shape)

    if test_classifier:	
        X_train,X_test,y_train, y_test = train_test_split(scaled_train_features,
                train_labels,test_size=test_ratio, random_state=42)

        trained_svm_clf = train_classifier(X_train,y_train)
        predictions = trained_svm_clf.predict(X_test)
        accuracy =  accuracy_score(y_test,predictions)

        print ('classifier accuracy: ', accuracy)
    else:
        trained_svm_clf = train_classifier(scaled_train_features,train_labels)

    return trained_svm_clf


        
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def process_test_images(clf):

      
    test_image_files =  glob.glob('./test_images/*.jpg')
    
    for img in test_image_files:
        test_img = cv2.imread(img)
        test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
        img_size = test_img.shape
        detection_bbox_list = []

        window_list1 = slide_window(test_img,y_start_stop=[int(img_size[0]*0.5),img_size[0]])
        search1_bbox = search_windows(test_img,window_list1,
                color_space= 'YCrCb',hog_channel='ALL')
        detection_bbox_list.append(search1_bbox)

        window_list2 = slide_window(test_img,y_start_stop=[int(img_size[0]*0.5),img_size[0]]
                ,xy_window=(32,32))
        search2_bbox = search_windows(test_img,window_list2,
                color_space= 'YCrCb',hog_channel='ALL')
        detection_bbox_list.append(search2_bbox)

        window_list3 = slide_window(test_img,y_start_stop=[int(img_size[0]*0.5),
            img_size[0]],xy_window=(16,16))
        search3_bbox = search_windows(test_img,window_list3,
                color_space= 'YCrCb',hog_channel='ALL')
        detection_bbox_list.append(search3_bbox) 

        search_result_img = draw_boxes(test_img,detection_bbox_list)
        

    



    return


if __name__ == '__main__':
    svm_clf = vehicle_detection_training()
    print('svm_clf', svm_clf)



