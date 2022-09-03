import cv2



image_height, image_width = 64, 64

max_images_per_class = 8000
dataset_directory = "UCF50"
classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
model_output_size = len(classes_list)



def extract_resize_normalize(video):
    
    frame_list = []
    video_reader = cv2.VideoCapture(video)
    while True:
        sucess, frame = video_reader.read()
        
        if not sucess:
            break
        frame = cv2.resize(frame, (image_height, image_width))
        
        frame = frame/255
        
        frame_list.append(frane)
        
    video_reader.release()
    return frame_list


