from  utils import create_dataset, load_processed_data, test_model
from train import trainer
from vis_utils import predict_on_live_video
import os


print("1: Create Dataset 2: Train Model 3: Test 4: Video Test")
while True:
  choice = int(input("Enter Your Choice :"))

  if choice ==1:
      create_dataset()
  if choice == 2:
      trainer()
  if choice ==3:

      test_model()
  if choice==4:
      print("Test on video file")

      filepath = os.getcwd()+'/UCF50/BaseballPitch/v_BaseballPitch_g01_c01.avi'
      outputpath = os.getcwd()+'Outputfiles/out1.mp4'
      window_size = 10

      predict_on_live_video(filepath, outputpath, window_size)
    
    
    
    
