#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
from utils import label_map_util
from utils import visualization_utils_color as vis_util

# 얼굴 인식 모델과 박스 레이블
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 얼굴인식 Detector 클래스 --> 메인 파일에 써줌
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
        """image: rgb 색상 이미지 
        return 값: (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지의 배열 기반 표현은 상자와 레이블이 있는 결과 이미지를 준비하기 위해 나중에 사용 
        # 이미지가 [1, None, None, 3] 모양을 가질 것으로 예상하므로 치수를 확장
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # 각 상자는 특정 물체가 감지된 이미지의 일부
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # 각 점수는 각 개체에 대한 신뢰 수준
        # 점수는 클래스 레이블과 함께 결과 이미지에 표시 --> 안쓰긴함
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        return (boxes, scores, classes, num_detections)

    
# ipynb에서 작성했기 때문에 겹치는 모듈 있을 수 있음
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
import matplotlib.pyplot as plt
from multiprocessing import Process, Value


font = cv2.FONT_ITALIC # 보여주기용

# 개체가 포함된 이미지  영역을 가져와 개체의 포즈를 정의하는 점 위치 집합을 출력하기 위한 predictor
predictor = dlib.shape_predictor("/Users/yujeong/Desktop/졸프용/shape_predictor_68_face_landmarks.dat")

tDetector = TensoflowFaceDector(PATH_TO_CKPT)



# 모델 경로
emotion_model_path = '/Users/yujeong/Desktop/CNN_test/tensorflow-face-detection-master/_mini_XCEPTION_model_korean_64_2.hdf5'

EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"] # 7가지 표정
ratio=[0,0,0,0,0,0] # 시각화를 위한 표정 비율
eye_fix=0 # 눈 흔들림 시각화를 위한 cnt
max_boxes_to_draw = 5

# 발표가 끝난 후 표정과 눈 흔들림 평균을 내기 위함  --> 최대 10명으로 하긴 했는데 5명으로 줄임
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

people_eye = [0,0,0,0,0,0,0,0,0,0]

emotion_classifier = load_model(emotion_model_path, compile=False)  # 표정 인식 모델 로드

img_path = os.getcwd()  # 현재 경로



# 표정인식 함수
def aaa(video_pause): 
    recording = False  # 녹화 기능 false로 초기화
    cap = cv2.VideoCapture(0)  # 비디오 켜기 
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 녹화
    
    # count
    eye_cnt=0
    frame_cnt=0
    
    start = time.time()  # 시작 시간 
    
    while cap.isOpened():
        status, frame = cap.read()

        if status:            
            frame_cnt = frame_cnt+1 # 프레임 수 세기 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백 전환 
            gray = cv2.GaussianBlur(gray, (7,7), 0) # 가우시안 블러 

            canvas = np.zeros((250, 300, 3), dtype="uint8") # 표정 분포 보여줄 화면 
            frameClone = frame.copy()  # 얼굴 보여줄 화면 


            # 녹화 상태 
            if recording:
                info = "Recording ON"
                out = cv2.VideoWriter('/Users/yujeong/Desktop/졸프용/SaveVideo.mp4',fourcc,20.0,(width, height))
                out.write(frame)

            else: # 녹화 안할 시 
                info = "Recording OFF"

            # 상단에 녹화 알림     
            cv2.putText(frameClone, info, (5,15), font, 0.5, (255,0, 255),1)


            (boxes, scores, classes, num_detections) = tDetector.run(gray)

            faces = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)


            people_cnt = -1

            for i in range(min(max_boxes_to_draw, faces.shape[0])): # 한 프레임에서 인식한 얼굴의 수만큼  // 5:

                if scores is None or scores[i] > 0.5:
                    people_cnt = people_cnt + 1
                    face = faces[i]  

                    # ymin, xmin, ymax, xmax = box
                    left = int(face[1] * width)
                    top = int(face[0] * height)
                    right = int(face[3] * width)
                    bottom = int(face[2] * height)


                    face = dlib.rectangle(left, top, right, bottom) 

                    landmarks = predictor(gray, face) # 랜드마크 따기 


                    # 얼굴 특징점
                    fX, fY = face.left(), face.top()
                    x1, y1 = face.right(), face.bottom()
                    fW = x1 - fX
                    fH = y1 - fY

                    # 얼굴 bounding box
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 0, 255), 1)


                    # 회색 이미지에서 얼굴의 ROI를 추출하고 고정된 28x28 픽셀로 크기를 조정한 다음 준비
                    # CNN분류에 대한 ROI
                    roi = gray[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)  

                    preds = emotion_classifier.predict(roi, verbose=0)[0] # 예측
                    label = EMOTIONS[preds.argmax()] # 가장 높은 비율을 label로 


                    # 왼쪽 눈 랜드마크 따서 좌표와 길이 구하기
                    left_x = landmarks.part(36).x
                    left_y = landmarks.part(37).y
                    left_w = landmarks.part(39).x - landmarks.part(36).x
                    left_h = landmarks.part(41).y - landmarks.part(37).y


                    # 왼쪽 눈에 대한 roi
                    left_roi_color = frameClone[left_y:left_y+left_h, left_x:left_x+left_w]
                    left_roi_gray = gray[left_y:left_y+left_h, left_x:left_x+left_w]

                    # 오른쪽 눈 랜드마크 따서 좌표와 길이 구하기
                    right_x = landmarks.part(42).x
                    right_y = landmarks.part(43).y
                    right_w = landmarks.part(45).x - landmarks.part(42).x
                    right_h = landmarks.part(47).y - landmarks.part(43).y

                    # 오른쪽 눈에 대한 roi
                    right_roi_color = frameClone[right_y:right_y+right_h, right_x:right_x+right_w]
                    right_roi_gray = gray[right_y:right_y+right_h, right_x:right_x+right_w]

                    # roi 크기 구하기 
                    left_rows, left_cols = left_roi_gray.shape
                    right_rows, right_cols = right_roi_gray.shape


                    # 왼쪽 동공 컨투어
                    # 50보다 크면 0(흑), 작으면 255(백)로 할당
                    _, threshold = cv2.threshold(left_roi_gray,50, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

                    # 오른쪽 동공 컨투어 
                    # 50보다 크면 0(흑), 작으면 255(백)로 할당
                    _, threshold2 = cv2.threshold(right_roi_gray, 50, 255, cv2.THRESH_BINARY_INV)
                    contours2, _ = cv2.findContours(threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours2 = sorted(contours2, key=lambda x: cv2.contourArea(x), reverse=True)


                    # 왼쪽 동공
                    for cnt in contours:
        #                 cv2.drawContours(left_roi_color, [cnt], 0, (255, 0, 0), 1)  # 컨투어 시각화  --> 꺼둠
                        (left_x,left_y,left_w,left_h) = cv2.boundingRect(cnt) # 컨투어 딴 눈동자에 외접하는 사각형

                        # 동공 직교 시각화  --> 꺼둠
    #                     cv2.line(left_roi_color, (left_x+int(left_w/2), 0), (left_x+int(left_w/2), left_rows), (255, 0, 0), 1) 
    #                     cv2.line(left_roi_color, (left_x, left_y+int(left_h/2)), (left_x+left_w, left_y+int(left_h/2)), (255,0,0), 1)

                        # 왼쪽 눈 시선 흐트러짐 감지 
                        if ((left_cols/2)-(left_cols*0.2) > left_x+left_w/2) or ((left_cols/2)+(left_cols*0.2) < left_x+left_w/2) and recording:
                            people_eye[people_cnt] = people_eye[people_cnt] + 1

                        break

                    # 오른쪽 동공
                    for cnt2 in contours2:
        #                 cv2.drawContours(right_roi_color, [cnt2], 0, (255, 0, 0), 1)  # 컨투어 시각화  --> 꺼둠
                        (right_x,right_y,right_w,right_h) = cv2.boundingRect(cnt2) # 컨투어 딴 눈동자에 외접하는 사각형

                        # 동공 직교 시각화  --> 꺼둠
    #                     cv2.line(right_roi_color, (right_x+int(right_w/2), 0), (right_x+int(right_w/2), right_rows), (255, 0, 0), 1)
    #                     cv2.line(right_roi_color, (right_x, right_y+int(right_h/2)), (right_x+right_w, right_y+int(right_h/2)), (255,0,0), 1)

                        # 오른쪽 눈 시선 흐트러짐 감지 
                        if ((right_cols/2)-(right_cols*0.2) > right_x+right_w/2) or ((right_cols/2)+(right_cols*0.2) < right_x+right_w/2) and recording:
                            people_eye[people_cnt] = people_eye[people_cnt] + 1

                        break



                # 감정 분포용 그래프 
                    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                            # construct the label text
                            if (label == "scared"):
                                label = "neutral"

                            text = "{}: {:.2f}%".format(emotion, prob * 100)

                            w = int(prob * 300)
                            cv2.rectangle(canvas, (6, (i * 35) + 5),
                            (w, (i * 35) + 35), (0, 0, 255), -1)
                            cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
                            cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


                            # 위치 별로 저장해두고 보여주기!!        

                            # prob가 0.6 이상일 때 (표정 비율 시각화를 위함)
                            if prob > 0.6: 
                                for i in range(len(EMOTIONS)):
                                    if emotion == EMOTIONS[i]:
                                        people_emo[people_cnt][i] = people_emo[people_cnt][i] + 1



                # 화면 띄우기 
#                 cv2.imshow('your_face', frameClone)  --> 끔(GUI 내의 캠이 있음)
    #             cv2.imshow("Probabilities", canvas)  --> 실시간 표정 비율
            
            
            

        if video_pause.value == 1:      # 발표 종료시 GUI에서 공유 변수   
#             print("video pause")
            recording = False
            end = time.time() 
            total_time = int(end - start)  # 총 시간 계산
            
            # 눈 움직인 비율 구하기 
            for i in range(0,people_cnt+1):
                people_eye[i] = 1 - (people_eye[i]/frame_cnt)
            
            # 사람 표정 그래프로 
            people_num = -1
            for ratio in people_emo:
                people_num = people_num + 1
                ratio_sum = 0
                for i in ratio:
                    ratio_sum = ratio_sum + i

                if(ratio_sum != 0):
                    plt.pie(ratio, labels=EMOTIONS, autopct='%.1f%%')
                    plt.savefig(img_path + '/save_fig/save_emotion/emotion%d.png'%people_num)
                    plt.clf() 

                else: break


            # 눈 흔들림 그래프로 
            cnt_eye_fig = -1
            for eye in people_eye:
                cnt_eye_fig = cnt_eye_fig+1

                if(eye != 0):
                    plt.pie([eye*100, (1-eye)*100], labels=["fix","not fix"], autopct='%.1f%%')
                    plt.savefig(img_path + '/save_fig/save_eye/eye%d.png'%cnt_eye_fig)
                    plt.clf() 
                    
                else: break

            break
            
#     cap.release()
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)        
