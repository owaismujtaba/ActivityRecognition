from matplotlib import pyplot as plt
import os
import pdb
from collections import deque
import cv2
import numpy as np

image_height, image_width = 64, 64

classes_list = ["Basketball", "BreastStroke", "GolfSwing", "MilitaryParade"]
def plot_train_accuracy_loss(history):
    #pdb.set_trace()
    epochs = [x for x in range(0, len(history.history))]
    plt.plot(epochs, history.history['accuracy'])
    plt.plot(epochs, history.history['loss'])
    plt.legend(['Accuracy', 'Loss'])
    plt.savefig(ps.getcwd()+'/Results/train_acc_loss.png', dpi=600)
    
    

    


def predict_on_live_video(video_file_path, output_file_path, window_size):
    
    
    from model import cnn_model
    from tensorflow.keras.models import save_model
    #pdb.set_trace()
    model = cnn_model()
    save_model(model, os.getcwd()+'/Models/cnnmodel.h5')
    
    
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

 

    # Writing the Overlayed Video Files Using the VideoWriter Object

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

    while True:

        # Reading The Frame

        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions

        resized_frame = cv2.resize(frame, (image_height, image_width))

         

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1

        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object

        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process

        if len(predicted_labels_probabilities_deque) == window_size:
            #pdb.set_trace()
            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

 

            # Calculating Average of Predicted Labels Probabilities Column Wise

            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
            # Converting the predicted probabilities into labels by returning the index of the maximum value.

            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
 
            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]

            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Writing The Frame
        video_writer.write(frame)
        # cv2.imshow('Predicted Frames', frame)
        # key_pressed = cv2.waitKey(10)
 
        # if key_pressed == ord('q'):

        #     break
 
    # cv2.destroyAllWindows()

     
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()
    print("Done")
    video_writer.release()
