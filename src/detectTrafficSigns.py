#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************************************
* OpenCV HAAR sign detection
* 20221026a
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
def detectTrafficSigns_GUI():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):
        confFileName = "detectTrafficSigns.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
            
    def paramDef(self):
        self.conf['Parameters'] = {
            'haarTrafficLight' : 'TrafficLight_HAAR_16Stages.xml',
            'haarStopSign' : 'stop_data.xml',
            'haarSpeedLimit' : 'Speedlimit_HAAR_ 17Stages.xml',
            'intervalStream' : 0.2,
            }
        
    def readConfig(self,configFile):
            try:
                self.conf.read(configFile)
                self.paramDef = self.conf['Parameters']
                self.haarTrafficLight = self.paramDef['haarTrafficLight']
                self.haarStopSign = self.paramDef['haarStopSign']
                self.haarSpeedLimit = self.paramDef['haarSpeedLimit']
                self.intervalStream = self.conf.getfloat('Parameters','intervalStream')
            except:
                print(" Error in reading configuration file. Please check it\n")
            
    # Create configuration file
    def createConfig(self):
        try:
            self.paramDef()
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
        print( "You need a runing instance of X Windows, Abort\n")
        return
    
    try:
        if self.isOpen:
            pass
    except:
        cam = Camera()
        
    top.geometry('800x700')
    top.title('Detect Traffic Signs')
    top.configure(background='#CDCDCD')

    filelabel=Label(top,background='#CDCDCD', font=('arial',15))
    sign_image = Label(top)
    label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
    chkBtnSign1 = tk.IntVar()
    chkBtnSign2 = tk.IntVar()
    chkBtnSign3 = tk.IntVar()
    
    def get_webcam_image():
        img = cam.getImage()
        process_Image(img)
        
    loop = RepeatedTimer(dP.intervalStream,get_webcam_image)
    
    def get_webcam_stream():
        loop.start()
                
    def process_Image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if chkBtnSign1.get() == 1:
            img_temp1 = signDetect(image, dP.haarStopSign, "Stop sign", 255,0,0)
        else:
            img_temp1 = image
        if chkBtnSign2.get() == 1:
            img_temp2 = signDetect(img_temp1, dP.haarSpeedLimit, "Speed Limit Sign", 0,255,0)
        else:
            img_temp2 = img_temp1
        if chkBtnSign3.get() == 1:
            img_temp3 = signDetect(img_temp2, dP.haarTrafficLight, "Traffic Light",0,0,255)
        else:
            img_temp3 = img_temp2
        img = img_temp3
            
        #width = int((img.shape[1]-img.shape[0])/2)
        #width2 = int((img.shape[1]+img.shape[0])/2)
        #uploaded = Image.fromarray(img[:,width:width2,:])
        uploaded = Image.fromarray(img)
        uploaded.thumbnail(((top.winfo_width()/1.2),(top.winfo_height()/1.2)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        
    def stop_stream():
        if loop.is_running:
            loop.stop()
        #cam.releaseCam()
    
    startStream=Button(top,text="Start camera stream",command=get_webcam_stream,padx=10,pady=5)
    startStream.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    #startStream.pack(side=BOTTOM,pady=50)
    startStream.place(relx=0.05,rely=0.20)
    
    stopStream=Button(top,text="Stop camera stream",command=stop_stream,padx=10,pady=5)
    stopStream.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    #stopStream.pack(side=BOTTOM,pady=50)
    stopStream.place(relx=0.05,rely=0.25)
    
    signChkBtn1 = tk.Checkbutton(top, text='Stop Sign Detection',variable=chkBtnSign1, onvalue=1, offvalue=0, command=get_webcam_stream)
    signChkBtn1.place(relx=0.70,rely=0.20)
    signChkBtn2 = tk.Checkbutton(top, text='US Speed Limits Sign Detection',variable=chkBtnSign2, onvalue=1, offvalue=0, command=get_webcam_stream)
    signChkBtn2.place(relx=0.70,rely=0.25)
    signChkBtn3 = tk.Checkbutton(top, text='Traffic Lights Detection',variable=chkBtnSign3, onvalue=1, offvalue=0, command=get_webcam_stream)
    signChkBtn3.place(relx=0.70,rely=0.30)
    
    sign_image.pack(side=BOTTOM,expand=True)
    label.pack(side=BOTTOM,expand=True)
    filelabel.pack(side=BOTTOM,expand=True)
    heading = Label(top, text="Detection of Traffic Signs",pady=5, font=('arial',20,'bold'))
    heading.configure(background='#CDCDCD',foreground='#364156')
    heading.pack()
    
    def on_closing():
        stop_stream()
        top.destroy()
            
    top.protocol("WM_DELETE_WINDOW", on_closing)
    top.mainloop()
    
#************************************
# Sign Detect
#************************************
def signDetect(img, typeSignalFile, label, c1, c2, c3):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    haar_data = cv2.CascadeClassifier(typeSignalFile)
    
    found = haar_data.detectMultiScale(img_gray,minSize =(20, 20))
    if len(found) !=0:
        print(" ",label,"found!")
        for (x, y, width, height) in found:
            cv2.rectangle(img, (x, y),
					(x + height, y + width),
					(c1, c2, c3), 5)
    else:
        #print(" NO",label,"sign")
        pass
    return img

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
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())




