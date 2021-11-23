import cv2
import numpy as np
import os

class FaceDetector():
    
    def __init__(self) :
        """
        カスケード分類器を作成
        """
        #カスケード分類器のパス取得
        self.baseDir = os.path.dirname(__file__)
        cascade_filename = 'haarcascade_frontalface_alt.xml'
        self.cascade_path = os.path.join(self.baseDir, cascade_filename)
        #カスケード分類器の読み込み
        self.cascade = cv2.CascadeClassifier(self.cascade_path)

    def detectFaceAreas(self, img):
        """
        顔の領域を検出する
        顔が複数ある場合も対応可能

        Parameters
        ----------
        img: 顔の領域を検出したい画像  
            
        Returns
        ----------
        face_areas(list): 顔の領域を格納（顔領域の、左端、上端、横方向の長さ、縦方向の長さ）
        """
        face_areas = self.cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        return face_areas
