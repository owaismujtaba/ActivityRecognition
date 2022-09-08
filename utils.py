import cv2
import os
import pdb
import random
import numpy as np


image_height, image_width = 64, 64

max_images_per_class = 8000
dataset_directory =  os.getcwd()+ "/UCF50/"
classes_list = ["Basketball", "BreastStroke", "GolfSwing", "MilitaryParade"]
model_output_size = len(classes_list)



def create_dataset():
    
    temp_features = []
    features = []
    labels = []
     
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')        
        # Getting the list of video files present in the specific class name directory

        files_list = os.listdir(os.path.join(dataset_directory, class_name))
        # Iterating through all the files present in the files list
        for file_name in files_list:
            # Construct the complete video path
            
            video_file_path = os.path.join(dataset_directory, class_name, file_name)
            
            # Calling the frame_extraction method for every video file path
            frames = extract_resize_normalize(video_file_path)
 
            # Appending the frames to a temporary list.
            temp_features.extend(frames)
        
        # Adding randomly selected frames to the features list
        features.extend(random.sample(temp_features, max_images_per_class))
 
        # Adding Fixed number of labels to the labels list
        labels.extend([class_index] * max_images_per_class)

         
        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()
 
    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels) 
        
    np.save(os.getcwd()+'/Data/features.npy', features)
    np.save(os.getcwd()+'/Data/labels.npy', labels)



def extract_resize_normalize(video):
    
    frame_list = []
    video_reader = cv2.VideoCapture(video)
    while True:
        sucess, frame = video_reader.read()
        
        if not sucess:
            break
        frame = cv2.resize(frame, (image_height, image_width))
        
        frame = frame/255
        
        frame_list.append(frame)
        
        
    video_reader.release()
    return frame_list

def load_processed_data():
    
    PATH = os.getcwd()+'/Data/'
    
    feature_path = PATH + 'features.npy'
    labels_path = PATH + 'labels.npy'
    
    features = np.load(feature_path)
    labels = np.load(labels_path)
    
    return features, labels


def test_model():
    from model import cnn_model
    from tensorflow.keras.models import save_model
    #pdb.set_trace()
    model = cnn_model()
    save_model(model, os.getcwd()+'/Models/cnnmodel.h5')
    x , y = load_processed_data()
    
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)
    
    from sklearn.metrics import accuracy_score
    
    print("Accuracy on the full dataset :", accuracy_score(pred, y))
    
    
    
        
