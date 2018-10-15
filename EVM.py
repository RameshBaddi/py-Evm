import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import time
import statistics
import math
import os
import subprocess
import sys
from flask import jsonify


#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=4):
    s=src.copy()
    pyramid=[s]
    for i in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid

#load video from file
def load_video(video_filename):
    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_tensor=np.zeros((frame_count,height,width,3),dtype='float')
    x=0
    while cap.isOpened():
        ret,frame=cap.read()
        if ret is True:
            video_tensor[x]=frame
            x+=1
        else:
            break
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    print("Temporal ideal filter", low, high)
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    print("gaussian_video", levels)
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i]=gaussian_frame
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification=10):
    print("Amplification", amplification)
    return gaussian_vid*amplification

#reconstruct video from original video and gaussian video
def reconstruct_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=img+origin_video[i]
        final_video[i]=img
    return final_video

#save video to files
def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("out.avi", fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

#magnify color
def magnify_color(video_name,low,high,levels=3,amplification=10):
    print ("Load video", low, high, levels, amplification)
    t,f=load_video(video_name)
    print ("Applying gaussian pyramid")
    gau_video=gaussian_video(t,levels=levels)
    print ("Applying temporal filter")
    filtered_tensor=temporal_ideal_filter(gau_video,low,high,f)
    print ("Amplifying filtered video")
    amplified_video=amplify_video(filtered_tensor,amplification=amplification)
    print ("Recontruct the video")
    final=reconstruct_video(amplified_video,t,levels=levels)
    print ("Saving the video")
    save_video(final)

# capture selfie video
def captureSelfieVideo(low=0.4, high=3, level=1, amplification=2):
    print ("Capturing selfie video start", low, high, level, amplification)
    cap = cv2.VideoCapture(0) # Starts captureing video. 0 -> To identify primary camera.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Define Codec for video saving.
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    if cap.isOpened()==False: # If the cap is not opened, open it.
        cap.open()

    videoWidth = cap.get(3) # Retrieve size of video. Same sould be set as framesize 
    videoHeight = cap.get(4)

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(videoWidth), int(videoHeight)))

    timeout = time.time() + 6 # 30 seconds from now.
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
            cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            if time.time() > timeout :
                break
        else:
            print("read function returning false")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    magnify_color("output.avi", low, high, level, amplification)

#end of captureSelfieVideo

def identifyFaceArea(video_filename):
    
    # Code to collect i-frames from the video.
        #command = 'ffmpeg -skip_frame nokey -i video_filename -vsync 0 -r 30 -f image2 thumbnails-%02d.jpeg'
        #out = subprocess.check_output(command + [video_filename]).decode()
        #f_types = out.replace('pict_type=','').split()
        #frame_types = zip(range(len(f_types)), f_types)
        #i_frames = [x[0] for x in frame_types if x[1]=='I']

    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    avg_brightness_per_frame=np.zeros((frame_count),dtype='float') # Stores average of the all pixel brightness for all the frames.
    m=0

    #Initialize a face cascade using the frontal face haar cascade provided
    #with the OpenCV2 library
    faceCascade = cv2.CascadeClassifier('/Users/administrator/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
    
    if cap.isOpened==False:
        cap.open()

    while frame_count!=0:
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame=cap.read()
        if ret is True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
            
            # To find the largest face in the frame/array of faces.
            maxArea = 0; x = 0; y = 0; w = 0; h = 0
            for (_x,_y,_w,_h) in faces:
                if  _w*_h > maxArea:
                    x = _x
                    y = _y
                    w = _w
                    h = _h
                    maxArea = w*h

            faceRect = frame[y: y+h, x: x+w]
            print (faceRect)
            avg_brightness_of_frame = faceRect.mean()
            
            
            avg_brightness_per_frame[m] = avg_brightness_of_frame #Storing the mean brightness of all the frames in array.
            m+=1
            
        frame_count-=1

    # Now we have average pixel brightness of all the frames.

    avg_brightness_per_frame = avg_brightness_per_frame[np.logical_not(np.isnan(avg_brightness_per_frame))]

    print (avg_brightness_per_frame)
    sd = statistics.stdev(avg_brightness_per_frame)
    print ("standard deviation %s", sd)

    cap.release()
    cv2.destroyAllWindows()
    return jsonify("Liveness", sd)
    
def findLiveness():
    #captureSelfieVideo(0.4, 4, 3)
    magnify_color("output.avi", 0.4, 4)
    sd = identifyFaceArea("out.avi")
    print(sd)
    return sd

if __name__=="__main__":
    #print("Input", sys.argv[1])
    #captureSelfieVideo(0.4, 4, 3)
    #magnify_color("output.avi", 0.4, 4)
    #identifyFaceArea("out.avi")
    findLiveness()
    






