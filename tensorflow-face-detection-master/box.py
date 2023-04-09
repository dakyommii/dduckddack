"""Webcam"""
from tkinter import Label
from PIL import Image, ImageTk
import cv2


class Box:

    def __init__(self, window, width=450, height=450):
        self.window = window
        self.width = width
        self.height = height

        self.label = Label(self.window, width=self.width, height=self.height)
        self.cap = cv2.VideoCapture(0)
        self.label.pack()


    def show_frames(self): # 화면에 보여주기 용
        """Show Frames"""
        # 젤 최근거 이미지로 변환
        cv2image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)

        imgtk = ImageTk.PhotoImage(image=img)  # Convert image --> PhotoImage

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        # 지속적으로 캡처하려면 일정 간격 후에 반복
        self.label.after(20, self.show_frames)
        
        
    def exit_cap(self):
        self.cap.release()  # --> 함수 만들긴 했지만 그냥 키고 있는 걸로 GUI 구성
