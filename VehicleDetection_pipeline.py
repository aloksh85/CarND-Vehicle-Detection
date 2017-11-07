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
import pickle
import os
from scipy.ndimage.measurements import label

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
    for img_file in  imgs:
        # Read in each one by one
        
        image = mpimg.imread(img_file)
        #image = image.astype(np.float32)/255.0
        #print('image: ',np.min(image),'-',np.max(image))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                        spatial_size=(32,32),
                        visualise=False,
                        spatial_features=True,
                        color_features=True):

    file_features =[]
    #print('pipeline parameters:')
    #print('orient: ',orient,
    #        'pix_per_cell: ',pix_per_cell,
    #        'cell_per_block: ',cell_per_block)
    #print('cspace: ',cspace)
    #print('spatial features: ',spatial_features)
    #print('color_features: ',color_features)
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
    
    if visualise:
        return np.concatenate(file_features),hog_image
    else:
        return np.concatenate(file_features)

def visualise_feature_image(cspace='YCrCb',orient=8,pix_per_cell=16,cell_per_block=2):
    
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

    car_img = mpimg.imread(cars[v_car_ind])
    #car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)
    #car_img= car_img.astype(np.float32)/255.0
    print('image: ',np.min(car_img),'-',np.max(car_img))
    
    notcar_img = mpimg.imread(notcars[v_notcar_ind])
    #notcar_img = cv2.cvtColor(notcar_img,cv2.COLOR_BGR2RGB)
    #notcar_img= notcar_img.astype(np.float32)/255.0
    print('image: ',np.min(notcar_img),'-',np.max(notcar_img))
    
    car_features1,carhog_image1 = single_img_features(car_img,
            cspace = cspace,hog_channel=0,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            visualise=True)
    notcar_features1,notcarhog_image1 = single_img_features(notcar_img,
            cspace=cspace,hog_channel= 0,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            visualise=True)

    car_features2,carhog_image2 = single_img_features(car_img,
            cspace = cspace,hog_channel=1,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            visualise=True)
    notcar_features2,notcarhog_image2 = single_img_features(notcar_img,
            cspace=cspace,hog_channel= 1,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            visualise=True)
    
    car_features3,carhog_image3 = single_img_features(car_img,
            cspace = cspace,hog_channel=2,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            visualise=True)
    notcar_features3,notcarhog_image3 = single_img_features(notcar_img,
            cspace=cspace,hog_channel= 2,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            visualise=True)
    
    f1,arr1 = plt.subplots(2,2,figsize=(15,15))
    arr1[0,0].imshow(car_img)
    arr1[0,1].imshow(np.hstack((carhog_image1,carhog_image2,carhog_image3)))
    arr1[1,0].imshow(notcar_img)
    arr1[1,1].imshow(np.hstack((notcarhog_image1,notcarhog_image2,notcarhog_image3)))
    f1.suptitle(cspace+', orient:'+str(orient)+', pixpercell: '
            +str(pix_per_cell)+ ', cellperblock: '+str(cell_per_block))

    


def train_classifier(train_features,train_labels,clf_type='SVM',svm_C=1.0):

    if clf_type =='SVM':
        print ('Training SVM')
        clf = train_SVM(train_features,train_labels,C_param =svm_C)
        print('Training complete.')

    return clf

    # split traiin test
    # train SVM
    # check accuracy
    # sliding window search
    # smoothing between frames
    #Train YOLO



def vehicle_detection_training(test_classifier=False,
                               color_feat=True,
                               spatial_feat=True,
                               svm_C=1.0,
                               test_ratio=0.3):
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

    car_img = mpimg.imread(cars[v_car_ind])
    #car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)

    not_car_img = mpimg.imread(notcars[v_notcar_ind])
    #not_car_img = cv2.cvtColor(not_car_img,cv2.COLOR_BGR2RGB)
    
    #visualise_feature_image(car_img,not_car_img,cspace='YCrCb')
    car_features = extract_features(cars,cspace='YCrCb',hog_channel='ALL',
            color_features=color_feat,spatial_features =spatial_feat)
    notcar_features = extract_features(notcars,cspace='YCrCb',hog_channel='ALL',
            color_features=color_feat,spatial_features=spatial_feat)
    
    train_labels = np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))
    train_features = np.vstack((car_features,notcar_features)).astype(np.float64)
    scaled_train_features,scaler = scale_features(train_features)

    print('scaled features shape: ',scaled_train_features.shape)
    print('labels vector shape: ', train_labels.shape)
    print('mean: ',np.mean(scaled_train_features),
            'std: ',np.std(scaled_train_features))

    if test_classifier:	
        X_train,X_test,y_train, y_test = train_test_split(scaled_train_features,
                train_labels,test_size=test_ratio, random_state=42)

        trained_svm_clf = train_classifier(X_train,y_train,svm_C=svm_C)
        predictions = trained_svm_clf.predict(X_test)
        accuracy =  accuracy_score(y_test,predictions)

        print ('classifier accuracy: ', accuracy)
    else:
        trained_svm_clf = train_classifier(scaled_train_features,train_labels,svm_C=svm_C)

    return trained_svm_clf,scaler


        
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    count = 0
    for window in windows:
        count+=1
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], 
            window[0][0]:window[1][0]],(64,64))

        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, cspace=color_space, 
                hog_channel='ALL',visualise = False,
                color_features=hist_feat,spatial_features=spatial_feat)
                           
        #print('img features shape: ',features.shape)

        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #print('scaled features shape: ',test_features.shape,
        #        'mean: ',np.mean(test_features),
        #        'std: ',np.std(test_features))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def process_test_images(clf,scaler,color_features=True,spatial_features=True):

      
    test_image_files =  glob.glob('./test_images/*.jpg')
    
    for img in test_image_files:
        f,arr = plt.subplots(1,2,figsize=(15,15))
        test_img = mpimg.imread(img)
        #test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
#        print('image: ',np.min(test_img),'-',np.max(test_img))
        test_img= test_img.astype(np.float32)/255.0
#        print('scaled image: ',np.min(test_img),'-',np.max(test_img))
        img_size = test_img.shape
        detection_bbox_list = []
        overlap = 0.5
        y_start_stop = [300,700]
        x_start_stop =[200,1280]
        
        window_list1=[]
        window_list1 = slide_window(test_img,xy_window=(64,64),
                y_start_stop=y_start_stop,xy_overlap=(0.5,0.5))
        
        window_list2=[]
        window_list2 = slide_window(test_img,x_start_stop=x_start_stop,
                y_start_stop= y_start_stop,xy_window=(96,96),
                xy_overlap=(0.5,0.5))
        window_list3=[]
        window_list3 = slide_window(test_img,x_start_stop=x_start_stop,
                y_start_stop=y_start_stop,
                xy_overlap=(0.5,0.5),xy_window=(80,80))
        
        hot_windows = search_windows(test_img,
                window_list3+window_list2+window_list1,
                clf,scaler, color_space= 'YCrCb',
                hog_channel='ALL',spatial_feat=spatial_features,               
                hist_feat=color_features)
        
        detection_bbox_list = detection_bbox_list+hot_windows
        
        draw_window_img = draw_boxes(test_img,hot_windows)
        heatmap = np.zeros_like(test_img[:,:,0]).astype(np.float)
        heatmap = add_heat(heatmap,hot_windows)
        heatmap = apply_threshold(heatmap,1)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(test_img), labels)        
        arr[0].imshow(test_img)
        arr[1].imshow(draw_img)
        

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
   # # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color=(255,0,0), thickness=6)
    # Return the image
    return img

def process_image(test_img,clf,scaler,hot_window_list,
        spatial_features=True,color_features=True):
        img_size = test_img.shape
        dup_img = np.copy(test_img.astype(np.float32)/255.0)
        #print('scaled image: ',np.min(test_img),'-',np.max(test_img))
        img_size = test_img.shape
        detection_bbox_list = []
        overlap = 0.5
        y_start_stop = [350,700]
        x_start_stop =[0,1280]
        
        window_list1=[]
        window_list1 = slide_window(dup_img,xy_window=(64,64),
                y_start_stop=y_start_stop,xy_overlap=(0.5,0.5))
        
        window_list2=[]
        window_list2 = slide_window(dup_img,x_start_stop=x_start_stop,
                y_start_stop= y_start_stop,xy_window=(96,96),
                xy_overlap=(0.5,0.5))
        window_list3=[]
        window_list3 = slide_window(dup_img,x_start_stop=x_start_stop,
                y_start_stop=y_start_stop,
                xy_overlap=(0.5,0.5),xy_window=(80,80))
        
        hot_windows = search_windows(dup_img,
                window_list3+window_list2+window_list1,
                clf,scaler, color_space= 'YCrCb',
                hog_channel='ALL',spatial_feat=spatial_features,               
                hist_feat=color_features)
        
        detection_bbox_list = detection_bbox_list+hot_windows
        
        #print('len hot windows: ',len(hot_windows))
        heatmap = np.zeros_like(test_img[:,:,0]).astype(np.float)
        hot_window_list.update_windows(hot_windows)
        #print('len all hot windows: ', len(hot_window_list.window_list()))
        draw_window_img = draw_boxes(test_img,hot_window_list.window_list())
        heatmap = add_heat(heatmap,hot_window_list.window_list())
        heatmap = apply_threshold(heatmap,2)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(test_img), labels)        
        
        return draw_img



# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def transformVideo(clip,clf,scaler,spatial_features,color_features,hot_window_list):
    temp_dir = "/home/alok/Documents/udacity_nd/temp_dir/"
    def image_transform(image):
        #transformVideo.count +=1
        #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        #cv2.imwrite(temp_dir+"img_"+str(transformVideo.count)+".jpg",image)

        return process_image(image,clf,scaler,
                hot_window_list,
                spatial_features,
                color_features)
    return clip.fl_image(image_transform)



def processVideo(videoPath,outputDir,clf,scaler,spatial_features,color_features):
    videoFileName = videoPath.split('/')[-1]
    print('video file name:',videoFileName)
    output = outputDir+'/out'+videoFileName
    print('out_video:',output)
    hot_window_list = HotWindows()
    clip  = VideoFileClip(videoPath)#.subclip(0,20)
    processed_clip = clip.fx(transformVideo,clf,scaler,
            spatial_features,
            color_features,
            hot_window_list)
    processed_clip.write_videofile(output,audio=False)



if __name__ == '__main__':

    if False:
        visualise_feature_image(cspace='HSV',orient=9,pix_per_cell=16,cell_per_block=2)
        visualise_feature_image(cspace='YCrCb',orient=9,pix_per_cell=8,cell_per_block=2)
        visualise_feature_image(cspace='LUV',orient=9,pix_per_cell=8,cell_per_block=2)
        visualise_feature_image(cspace='HLS',orient=9,pix_per_cell=8,cell_per_block=2)
        visualise_feature_image(cspace='RGB',orient=9,pix_per_cell=8,cell_per_block=2)
        plt.show()
    
    if False:
        svm_clf = None
        scaler = None
        spatial_features = True
        color_features = True
        orient=9
        pix_per_cell=8
        cell_per_block =2
        hist_bins =32
        spatial_size=(32,32)
        svm_C=0.01
        video =True
        cspace ='YCrCb'
        videopath="./project_video.mp4"
        output_dir ="./output_video"

        if os.path.isfile("vehicle_classifier_allfeats.p"):
            print("Loading classifer from file")
            classifier_dict = pickle.load(open("vehicle_classifier_allfeats.p","rb"))
            svm_clf = classifier_dict['classifier']
            scaler = classifier_dict['feature_scaler']
        else:
            svm_clf,scaler = vehicle_detection_training(test_classifier=False,
                    color_feat=color_features,
                    spatial_feat=spatial_features,svm_C=svm_C)
            classifier_dict ={'classifier':svm_clf,'feature_scaler':scaler}
            pickle.dump(classifier_dict,open("vehicle_classifier_allfeats.p","wb"))
           

        if video:
            processVideo(videopath,output_dir,svm_clf,scaler,spatial_features,color_features)
        else:
            process_test_images(svm_clf,scaler,
                spatial_features=spatial_features,color_features=color_features)
            plt.show()

