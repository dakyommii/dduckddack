from __future__ import division

import io

import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r"/Users/yujeong/Downloads/gradstt-ac34c3b32796.json"


# In[2]:


#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2


from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[3]:


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
#         start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
#         elapsed_time = time.time() - start_time
#         print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


# In[4]:


from keras.models import load_model
from tensorflow.keras.utils import img_to_array


import imutils
import cv2
import numpy as np
import dlib 
import datetime
import math 
import time 
import sys


# In[5]:




font = cv2.FONT_ITALIC

# 개체가 포함된 이미지  영역을 가져와 개체의 포즈를 정의하는 점 위치 집합을 출력하기 위한 predictor
predictor = dlib.shape_predictor("/Users/yujeong/Desktop/졸프용/shape_predictor_68_face_landmarks.dat")



tDetector = TensoflowFaceDector(PATH_TO_CKPT)



# 모델 경로
# detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml' # 안면 인식
# emotion_model_path = '/Users/yujeong/Downloads/_mini_XCEPTION.gitcode (2).hdf5' # 표정 인식
emotion_model_path = '/Users/yujeong/Downloads/_mini_XCEPTION_model_korean_64_2.hdf5'

EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"] # 7가지 표정
ratio=[0,0,0,0,0,0] # 시각화를 위한 표정 비율
eye_fix=0 # 눈 흔들림 시각화를 위한 cnt
max_boxes_to_draw = 5


people_emo = [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]


# In[6]:



# [START speech_transcribe_streaming_mic]
# from __future__ import division

import re
import sys

from google.cloud import speech

import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

f=open("stt.txt",'w')


# In[1]:


import time
import random
from multiprocessing import Process, Value, Pipe
import multiprocessing as mp


from untitled2 import faceDetect

from gptTest import main2, kogpt2


 # 공유 메모리를 이용한 멀티 프로세싱 
import warnings
warnings.filterwarnings('ignore')
        
from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter_webcam import webcam


def openFrame(frame):
    frame.tkraise()

    
def keyEvent():
    f = open('/Users/yujeong/Desktop/keywords.txt', 'r')
    s2 = f.readline()        
    keywordwrite['text'] = str(s2)
    
    
def event():
    strtBtn['text']='발표중'

def tailEvent():
#         sketchbook.delete(tail)
    f3 = open('/Users/yujeong/Desktop/question.txt', 'r')

    s3 = f3.readline()
    tailwrite['text'] = str(s3)


if __name__ == '__main__':    

#     ####################
    win=tk.Tk()
    win.title("뚝딱")
    win.geometry("1000x1000")
    win.configure(background='thistle1')

    mainFrm=tk.Frame(win)
    resFrm=tk.Frame(win)

    mainFrm.grid(row=0,column=0,sticky='nsew')
    resFrm.grid(row=0,column=0,sticky='nsew')
    
#     def openFrame(frame):
#         frame.tkraise()

    # 메인 화면
    # 웹캠
    mf1=Frame(mainFrm)
    mf1.pack(side="left")

#     video = webcam.Box(mf1, height=600,width=700)
#     video.show_frames()  # Show the created Box

    # 키워드 result
    mf2_1=Frame(mainFrm,background='white')
    mf2_1.pack()


    keyword=Label(mf2_1, text='추출 keyword') # fg는 글자 색 지정, font로 글자 설정
    keyword.place(x=140, y=10)
    keyword.pack()

#     keyword = sketchbook.create_text((140, 10), text="추출 keyword")

#     keywordwrite = sketchbook.create_text((140, 35), text="추출된 키워드 표시")

#     sketchbook.create_rectangle( 12, 20, 280, 200, outline = "purple4", width = "3")

    # 키워드추출 버튼
#     mf2_2=Frame(mainFrm,background='white')
#     mf2_2.pack()

#     def keyEvent():
#         f = open('/Users/yujeong/Desktop/keywords.txt', 'r')

#         s2 = f.readline()        
#         keywordwrite['text'] = str(s2)

    
#     keywordwrite = mf2_1.create_text((140, 35), text="추출된 키워드 표시")
    keywordwrite=Label(mf2_1, text='추출된 keyword') # fg는 글자 색 지정, font로 글자 설정
    keywordwrite.place(x=140, y=100)
    keywordwrite.pack()
    
    
    
    kwBtn=Button(mf2_1,text='키워드 추출하기',padx=10,pady=10,command=keyEvent)
    kwBtn.pack()

    # 꼬리질문 result
#     mf2_3=Frame(mainFrm,background='white')
#     mf2_3.pack()

    tail=Label(mf2_1, text='꼬리질문') # fg는 글자 색 지정, font로 글자 설정
    tail.place(x=140, y=200)
    tail.pack()
    
    
    tailwrite=Label(mf2_1, text='생성된 꼬리질문') # fg는 글자 색 지정, font로 글자 설정
    tailwrite.place(x=140, y=250)
    tailwrite.pack()
    # keyword = sketchbook.create_text((140, 100), text=" keyword")
    # keywordBtn=Button(mf2,text='키워드버튼',padx=10,pady=10)
#     tail=sketchbook.create_text((140, 10), text="꼬리질문")

    # sketchbook.create_rectangle( 12, 80, 280, 200, outline = "purple4", width = "3")
#     sketchbook.create_rectangle( 12, 20, 280, 300, outline = "magenta4", width = "3" )

    # 버튼
    # 발표종료  
    mf3=Frame(mainFrm)
    mf3.pack()

#     def event():
#         strtBtn['text']='발표중'

#     def tailEvent():
# #         sketchbook.delete(tail)
#         f3 = open('/Users/yujeong/Desktop/question.txt', 'r')

#         s3 = f3.readline()
#         tailwrite['text'] = str(s3)
# #         sketchbook.create_text((140, 320), text=s3)

        
        
        
#     queswrite=Label(mf2_1, text='꼬리 질문') # fg는 글자 색 지정, font로 글자 설정
#     queswrite.place(x=140, y=350)
#     queswrite.pack()

        
    strtBtn=Button(mf3,text='발표 시작',padx=10,pady=10,command=event)
    endBtn=Button(mf3,text='발표 종료',padx=10,pady=10,command=tailEvent)
    # 다음페이지로 화면 전환
    btnToRes=Button(mf3,text='결과 보기',padx=10,pady=10,command=lambda:[openFrame(resFrm)])  

    strtBtn.pack(side="left")
    btnToRes.pack(side="right")
    endBtn.pack(side="right")
    
    
    
    
    # 결과 화면
    # face Detction 결과
    rf1=Frame(resFrm)
    rf1.pack()

    sketchbook1 = Canvas(rf1,width=1000,height=205,background='white')
    sketchbook1.pack()

    faceDetect = sketchbook1.create_text((60, 20), text="표정인식 결과")
    sketchbook1.create_rectangle( 20, 30, 980, 200, outline = "purple4", width = "3")

    # 발화 속도 결과
    rf2=Frame(resFrm)
    rf2.pack()

    sketchbook2 = Canvas(rf2,width=1000,height=205,background='white')
    sketchbook2.pack()

    speed=sketchbook2.create_text((60, 20), text="발화속도 결과")
    sketchbook2.create_rectangle( 20, 30, 980, 200, outline = "magenta4", width = "3" )

    # 시선 처리 결과
    rf3=Frame(resFrm)
    rf3.pack()

    sketchbook3 = Canvas(rf3,width=1000,height=205,background='white')
    sketchbook3.pack()

    eye=sketchbook3.create_text((60, 20), text="시선처리 결과")
    sketchbook3.create_rectangle( 20, 30, 980, 200, outline = "magenta4", width = "3" )

    # 버튼
    rf4=Frame(resFrm)
    rf4.pack()

    btnToMain=Button(rf4,text="메인으로 돌아가기",padx=10,pady=10,command=lambda:[openFrame(mainFrm)])

    btnToMain.pack()
    
    openFrame(mainFrm)

    
    ###################3
    
    face_func = Process(target=faceDetect)
    face_func.start()

#     nlp_func = Process(target=main2)
#     nlp_func.start()

#     face_func = Process(target=faceDetect)
    loop_func = Process(target=win.mainloop())
    loop_func.start()

#     face_func.start()
    face_func.join()
#     nlp_func.join()
#     face_func.join()

    loop_func.join()
    
    


# In[ ]:




