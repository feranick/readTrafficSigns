#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************************************
* OpenCV keras traffic sign recognition - GUI prediction
* 20230217a
* https://data-flair.training/blogs/python-project-traffic-signs-recognition/
*****************************************************************************
'''
print(__doc__)

import tkinter as tk
import numpy as np
import scipy
import os, sys, platform, configparser, time, cv2, threading
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

#***************************************************
# This is needed for installation through pip
#***************************************************
def readTrafficSigns_GUI():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):
        confFileName = "readTrafficSigns_GUI.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"

        if platform.system() == 'Linux':
            self.edgeTPUSharedLib = "libedgetpu.so.1"
        if platform.system() == 'Darwin':
            self.edgeTPUSharedLib = "libedgetpu.1.dylib"
        if platform.system() == 'Windows':
            self.edgeTPUSharedLib = "edgetpu.dll"
            
    def rTSDef(self):
        self.conf['Parameters'] = {
            'name' : "roadSign",
            'intervalStream' : 0.2,
            }
        
    def sysDef(self):
        self.conf['System'] = {
            'useTFlitePred' : False,
            'TFliteRuntime' : False,
            'runCoralEdge' : False,
            }

    def readConfig(self,configFile):
            try:
                self.conf.read(configFile)
                self.rTSDef = self.conf['Parameters']
                self.sysDef = self.conf['System']
                self.name = self.rTSDef['name']
                self.intervalStream = self.conf.getfloat('Parameters','intervalStream')
                self.useTFlitePred = self.conf.getboolean('System','useTFlitePred')
                self.TFliteRuntime = self.conf.getboolean('System','TFliteRuntime')
                self.runCoralEdge = self.conf.getboolean('System','runCoralEdge')
            except:
                print(" Error in reading configuration file. Please check it\n")
            
    # Create configuration file
    def createConfig(self):
        try:
            self.rTSDef()
            self.sysDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#************************************
# Main
#************************************
def main():
    dP = Conf()

    #initialise GUI
    try:
        top=tk.Tk()
    except:
        print( "You need a running instance of X Windows, Abort\n")
        return
    
    try:
        if self.isOpen:
            pass
    except:
        cam = Camera()
        
    top.geometry('800x700')
    top.title('Traffic sign classification')
    top.configure(background='#CDCDCD')

    filelabel=Label(top,background='#CDCDCD', font=('arial',15))
    sign_image = Label(top)
    label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
    
    model = loadModel()

    def upload_image():
        file_path=filedialog.askopenfilename()
        img = cv2.imread(file_path)
        process_Image(img)
           
    def get_webcam_image():
        img = cam.getImage()
        process_Image(img)
        
    loop = RepeatedTimer(dP.intervalStream,get_webcam_image)
    
    def get_webcam_stream():
        loop.start()
                
    def process_Image(img):
        width = int((img.shape[1]-img.shape[0])/2)
        width2 = int((img.shape[1]+img.shape[0])/2)
        uploaded = Image.fromarray(img[:,width:width2,:])
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        filelabel.configure(text='')
        try:
            classify(uploaded)
        except:
            print("Classification failed")
            
    def classify(image):
        #filelabel.configure(foreground='#011638', text=file_path)
        image = image.resize((30,30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        #pred = model.predict_classes([image])[0]
        #pred = np.argmax(model.predict([image]), axis=-1)[0]
        
        predictions, probabilities = getPredictions(image, model)
        pred_class = np.argmax(predictions, axis=-1)
        pred = pred_class[0]
        prob = 100*predictions[0][pred_class]
        
        if dP.useTFlitePred:
            prob = prob/255
                        
        sign = classes()[pred+1]
        #print(" Sign:\033[1m",sign,"\033[0m - File:",file_path)
        print(" Sign:\033[1m",sign,"\033[0m\t\tProbability: ",np.round(prob,2)[0], \
            " (",probabilities[0][pred_class][0],")")
        
        label.configure(foreground='#011638', text=sign)
        
    def stop_stream():
        if loop.is_running:
            loop.stop()
        #cam.releaseCam()

    upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
    upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    #upload.pack(side=BOTTOM,pady=50)
    upload.place(relx=0.05,rely=0.25)

    startCamera=Button(top,text="Start camera",command=get_webcam_image,padx=10,pady=5)
    startCamera.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    #startCamera.pack(side=BOTTOM,pady=50)
    startCamera.place(relx=0.05,rely=0.20)
    
    startStream=Button(top,text="Start camera stream",command=get_webcam_stream,padx=10,pady=5)
    startStream.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    #startStream.pack(side=BOTTOM,pady=50)
    startStream.place(relx=0.70,rely=0.20)
    
    stopStream=Button(top,text="Stop camera stream",command=stop_stream,padx=10,pady=5)
    stopStream.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    #stopStream.pack(side=BOTTOM,pady=50)
    stopStream.place(relx=0.70,rely=0.25)
    
    sign_image.pack(side=BOTTOM,expand=True)
    label.pack(side=BOTTOM,expand=True)
    filelabel.pack(side=BOTTOM,expand=True)
    heading = Label(top, text="Traffic Sign Identification",pady=5, font=('arial',20,'bold'))
    heading.configure(background='#CDCDCD',foreground='#364156')
    heading.pack()
    
    def on_closing():
        stop_stream()
        top.destroy()
            
    top.protocol("WM_DELETE_WINDOW", on_closing)
    top.mainloop()
    
#************************************
# Load saved models
#************************************
def loadModel():
    dP = Conf()
    if dP.TFliteRuntime:
        import tflite_runtime.interpreter as tflite
        # model here is intended as interpreter
        if dP.runCoralEdge:
            print(" Running Tensorflow lite on Coral Edge TPU\n")
            try:
                model = tflite.Interpreter(model_path=dP.name+'_model_edgetpu.tflite',
                    experimental_delegates=[tflite.load_delegate(dP.edgeTPUSharedLib,{})])
            except:
                print(" Coral Edge TPU not found or compiled model not found. \n")
        else:
            print(" Using Tensoflow lite runtime\n")
            model = tflite.Interpreter(model_path=dP.name+'_model.tflite')
        model.allocate_tensors()
    else:
        #getTFVersion(dP)
        import tensorflow as tf
        if dP.useTFlitePred:
            print("Using Tensorflow lite\n")
            # model here is intended as interpreter
            model = tf.lite.Interpreter(model_path=dP.name+'_model.tflite')
            model.allocate_tensors()
        else:
            model = tf.keras.models.load_model(dP.name+'_model_classifier.h5')
    return model

#************************************
# Make prediction based on framework
#************************************
def getPredictions(R, model):
    dP = Conf()
    if dP.useTFlitePred:
        interpreter = model  #needed to keep consistency with documentation
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        #input_data = np.array(R*255, dtype=np.uint8) # Disable this for TF1.x
        input_data = np.array(R, dtype=np.uint8) # Disable this for TF1.x
        #input_data = np.array(R, dtype=np.float32)  # Enable this for TF2.x (not compatible with on EdgeTPU)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        predictions = interpreter.get_tensor(output_details[0]['index'])
        #probabilities = tf.math.softmax(predictions.astype('double')).numpy()
        
    else:
        predictions = model.predict(R)
        
    probabilities = scipy.special.softmax(predictions.astype('double'))
    return predictions, probabilities

#************************************
# Get image from webcam
#************************************
class Camera():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.isOpen = True
    
    def releaseCam(self):
        self.cam.release()
        cv2.destroyAllWindows()
    
    def getImage(self):
        self.check, self.frame = self.cam.read()
        return self.frame
        
    def destroy():
        self.cam.release()
    
#************************************
# Repeating method
#************************************

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        #self.start()  #Enable for starting upon initialization

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

#************************************
# Define classes of labels
#************************************
def classes():
    #dictionary to label all traffic signs class.
    classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
            10:'No passing',
            11:'No passing veh over 3.5 tons',
            12:'Right-of-way at intersection',
            13:'Priority road',
            14:'Yield',
            15:'Stop',
            16:'No vehicles',
            17:'Veh > 3.5 tons prohibited',
            18:'No entry',
            19:'General caution',
            20:'Dangerous curve left',
            21:'Dangerous curve right',
            22:'Double curve',
            23:'Bumpy road',
            24:'Slippery road',
            25:'Road narrows on the right',
            26:'Road work',
            27:'Traffic signals',
            28:'Pedestrians',
            29:'Children crossing',
            30:'Bicycles crossing',
            31:'Beware of ice/snow',
            32:'Wild animals crossing',
            33:'End speed + passing limits',
            34:'Turn right ahead',
            35:'Turn left ahead',
            36:'Ahead only',
            37:'Go straight or right',
            38:'Go straight or left',
            39:'Keep right',
            40:'Keep left',
            41:'Roundabout mandatory',
            42:'End of no passing',
            43:'End no passing veh > 3.5 tons' }
    return classes


#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())




