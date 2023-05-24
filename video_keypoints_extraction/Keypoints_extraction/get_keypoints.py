import argparse
import os
import time
import pickle
import numpy as np
import pandas as pd
from Feature_extraction_RawCode import get_video_features

#This represents the body parts to compare, more info in the ReadMe
parts_to_compare = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15),
                    (14, 16)]

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model_path", help="path to the tflite model",
                    default="../models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
parser.add_argument('--video_dir', type=str, default='../../data/videos/' ) 

# Read arguments from the command line
args = parser.parse_args()





if __name__ == '__main__':
    
    starting_time = time.time()
    
    dataset_angles = dict()
    dataset_coords = dict()
    
    #Getting the paths of all videos 
    all_videos_paths = [
        f.path for f in os.scandir(args.video_dir) if f.is_file() and f.path.endswith(('.mov','.mp4'))]
        
    print("\n","Starting computation....")
    for index,path in enumerate(all_videos_paths):
        
        angles, coordinates = get_video_features(args.model_path, path)
        video_name = os.path.splitext(os.path.basename(path))[0]
        
        angles = np.array(angles)
        coordinates = np.array(angles)
        
        #--Addin cosines and sines of angles
        angles = np.hstack([angles,np.cos(angles*np.pi/180),np.sin(angles*np.pi/180)])
        
        print("\n",f"...Dealing with video {video_name} \n")
        
        dataset_angles.update({video_name:angles})
        dataset_coords.update({video_name:coordinates})
        
        #--Creating csv_files
        df = pd.DataFrame(angles)
        df.to_csv(f"../{video_name}_rot.csv")
        
    with open('../outputFile_angles.pickle','wb') as outputFile :
        pickle.dump(dataset_angles,outputFile)
    
    with open('../outputFile_coords.pickle','wb') as outputFile :
        pickle.dump(dataset_coords,outputFile)
        
    print("Time to compute:", round(time.time() - starting_time),"seconds\n")   
    
