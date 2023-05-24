# DeepMotionism

### License
The DeepMotionism is licensed under the terms of the MIT license.

### Related Article

> Hou, Y., Seydou, F.M. and Kenderdine, S. (2023), "Unlocking a multimodal archive of Southern Chinese martial arts through embodied cues", Journal of Documentation. [https://doi.org/10.1108/JD-01-2022-0027](https://doi.org/10.1108/JD-01-2022-0027) 

> Or, see preprint on [ResearchGate](https://www.researchgate.net/publication/370497096_Unlocking_a_multimodal_archive_of_Southern_Chinese_martial_arts_through_embodied_cues)


---


### Introduction to directories

- "./docs/" contains an interactive documentation of the source codes. Launch it by opening "./docs/build/html/index.html" and explore the content in your browser .


- "./src/" contains all the source codes (python scripts) developed for this project.

- "./data/datasets/" contains : 
	1. "datasetHighVariance" : the preprocessed mocap data where the features with highest variance have bee selected.
	2. "motion_words" : the motion words extracted from all the joint-rotation conversions of the mocap data available at "./data/csv/"
	3. training_dataset_5 : the training dataset where 5 positive and negative examples have been selected for each motion word.

- "./data/bvh/" contains all the bvh files considered for this project

- "./data/csv/" contains all the joint-rotation converion of the bvh files stated above.
	

### Data preprocessing

-- .bvh files parsing :
    - The MoCap data in bvh format have been parsed using the python library "bvhtoolbox". We extracted the joint rotation angle that we save in csv files.
      There are guidelines on how to use this library at https://github.com/OlafHaag/bvh-toolbox .

-- Keypoints extraction from videos :
    - The videos are provided in mp4 and mov format. To extract the joint angles of the person performing in the videos, we use a PoseNet model and then manually compute the joint rotation angles.
      To extract the keypoints, put all videos in "/data/videos/", then head to "/video_keypoints_extraction/Keypoints_extraction/" and finally run "get_keypoints.py". 
      All csv files containing joint rotation angle will be put into "/video_keypoints_extraction/". So from there one can choose to put them in "/data/csv/Train" or "/data/csv/Test".
    - Our model assume that there is only one person in the video ! If there are multiple people then one should think about using "OpenPose" to extract the keypoints and then convert it to a bvh files as stated at https://github.com/Shimingyi/MotioNet .
    
    
### Requirements
Simply run "pip install -r requirements.txt" to get all the libraries and their dependencies necessary to run the code.


### Running the GUI
To run the GUI, simply run this command "streamlit run gui.py" from the command line after "cd ./src/".
The GUI is quite straightforward and allows to play the videos

