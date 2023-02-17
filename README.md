# readTrafficSigns

readTrafficSings uses CNN (which can be trained with included script) to classify European traffic signs. 
detectTrafficSigns uses HAAR algorithms for detecting a limited set of traffic signs (stop, traffic lights, speed limits). 

Features
=============
1. Prediction using uploaded images or from webcam
2. Prediction on live video streams (rate > 0.2 s) from camera
3. Stop Sign detection and direct visualization
2. Save models
3. Decoupled train/predict (with GUI for the latter)
4. Support for embedded platforms (Coral/EdgeTPU, tensorflow lite)
5. Installable setup.py
6. INI file for configuration parameters (no need to change the source once installed)

Required libraries
===================
   - tensorflow (>= 2.12.x)
   - numpy
   - scikit-learn (>=0.18)
   - scipy
   - matplotlib
   - pandas
   - h5py
   - Optional: tensorflow-lite (v.2.3 and higher)
   - Optional: [tensorflow-lite runtime](https://www.tensorflow.org/lite/guide/python) 
   - Optional: tensorflow-lite runtime with [Coral EdgeTPU](https://coral.ai/docs/accelerator/get-started/)
   
   In addition, these packages may be needed depending on your platform (via ```apt-get``` in debian/ubuntu or ```port``` in OSX):
    `python3-tk`

Credits
=============
Based on the code available here:
https://data-flair.training/blogs/python-project-traffic-signs-recognition/
Train/Validation/Test files available here:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Haar based detection: 
https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/

Additional haar detection files
https://github.com/cfizette/road-sign-cascades


===================================================
Deprecated code (old):
https://www.geeksforgeeks.org/opencv-and-keras-traffic-sign-classification-for-self-driving-car/
Train/Validation/Test files available here:
https://bitbucket.org/jadslim/german-traffic-signs/raw/a11dc223e3905f459e33abdb86673730e1e78509/test.p
https://bitbucket.org/jadslim/german-traffic-signs/raw/a11dc223e3905f459e33abdb86673730e1e78509/train.p
https://bitbucket.org/jadslim/german-traffic-signs/raw/a11dc223e3905f459e33abdb86673730e1e78509/valid.p
