# DeepMotionism

To allow users to more easily interact with our code, we have provided a sphinx documentation which is an interactive documentation that can be visualize in a browser.
In order to launch it, simply double click on "Documentation.lnk" in the /Doc.

**DeepMotionism is licensed under the MIT License**

## Related Article

> Hou, Y., Seydou, F.M. and Kenderdine, S. (2023), "Unlocking a multimodal archive of Southern Chinese martial arts through embodied cues", Journal of Documentation, Vol. ahead-of-print No. ahead-of-print. [https://doi.org/10.1108/JD-01-2022-0027](https://doi.org/10.1108/JD-01-2022-0027)

## Description of directories

-- In the directory "./data/bvh/" : 
	- We have all the bvh files that are being considered for this project

-- In the directory "./data/csv/" : 
	-  We have all the joint-rotation converion of the bvh files stated above.

-- In the directory "./data/datasets/" we have : 
	- "datasetHighVariance" : the preprocessed mocap data where the features with highest variance have bee selected.
	- "motion_words" : the motion words extracted from all the joint-rotation conversions of the mocap data available at "./data/csv/"
	- training_dataset_5 : the training dataset where 5 positive and negative examples have been selected for each motion word.

-- In the directory "./docs/":
	- We have the documentation of the project.
	- To visualize it, open the file at "./docs/build/html/index.html".

-- In the directory "./src/":
	- We have all the source python scripts used for the project.
	
## Data preprocessing

-- .bvh files parsing :
    - The MoCap data in bvh format have been parsed using the python library "bvhtoolbox". We extracted the joint rotation angle that we save in csv files.
      There are guidelines on how to use this library at https://github.com/OlafHaag/bvh-toolbox .

-- Keypoints extraction from videos :
    - The videos are provided in mp4 and mov format. To extract the joint angles of the person performing in the videos, we use a PoseNet model and then manually compute the joint rotation angles.
      To extract the keypoints, put all videos in "/data/videos/", then head to "/video_keypoints_extraction/Keypoints_extraction/" and finally run "get_keypoints.py". 
      All csv files containing joint rotation angle will be put into "/video_keypoints_extraction/". So from there one can choose to put them in "/data/csv/Train" or "/data/csv/Test".
    - Our model assume that there is only one person in the video ! If there are multiple people then one should think about using "OpenPose" to extract the keypoints and then convert it to a bvh files as stated at https://github.com/Shimingyi/MotioNet .
    
    
## Requirements
Simply run "pip install -r requirements.txt" to get all the libraries and their dependencies necessary to run the code.


## Running the GUI
To run the GUI, simply run this command "streamlit run gui.py" from the command line after "cd ./src/".
The GUI is quite straightforward and allows to play the videos

