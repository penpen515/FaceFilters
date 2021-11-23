import cv2
import sys
import os

sys.path.append(os.path.dirname(__file__))
from face_detector import FaceDetector

class FaceFilter():
    """
    矩形で検出した顔領域すべてに対してモザイクやぼかしなどのフィルタを適用
    
    Attributes
    ---------
    ratio[float]:　モザイクの強さ。数字が小さいほど強い
    kernel_size[int]:　ぼかしの強さ。一般的に奇数で数字が大きいほどぼかしが強い
    """
    ratio: float = 0.04
    kernel_size: int = 13
    
    def __init__(self):
        self.fd = FaceDetector()
        self.img_idx = 0


    def fiterPreProcessing(self, img,  x_left: int, y_top: int, x_right: int, y_bottom: int):
        """
        各フィルタを適用させる前に行う前処理
                
        Parameters
        ----------
        img: フィルタを適用させたい画像
        x_left(int): 顔領域の左端
        y_top(int): 顔領域の上端
        x_right(int): 顔領域の右端
        y_bottom(int): 顔領域の下端
        
        Retruns
        ----------
        face_img: 顔領域だけを切り抜いた画像
        x_left(int): 顔領域の左端
        y_top(int): 顔領域の上端
        x_right(int): 顔領域の右端
        y_bottom(int): 顔領域の下端
        """
        self.img_idx += 1
        face_img = img.copy()
        face_img = face_img[y_top:y_bottom, x_left:x_right]

        #検出された顔の大きさが画像サイズを超えた場合は、超える境界値までの範囲でフィルタを適用
        if (y_top < 0)               : y_top = 0
        if (y_bottom >= img.shape[0]): y_bottom = img.shape[0]-1
        if (x_left < 0)              : x_left = 0
        if (x_right >= img.shape[1]) : x_right = img.shape[1]-1
        
        return face_img, x_left, y_top, x_right, y_bottom
    
    def mosaicFilter(self, img, x_left: int, y_top: int, x_right: int, y_bottom: int):
        """
        検出した顔領域にモザイクをかけた画像を返す
        
        Parameters
        ----------
        img: フィルタを適用させたい画像
        x_left(int): 顔領域の左端
        y_top(int): 顔領域の上端
        x_right(int): 顔領域の右端
        y_bottom(int): 顔領域の下端
        
        Retruns
        ----------
        mosaic_img: imgの顔領域にモザイクを適用した画像
        """
        face_img, x_left, y_top, x_right, y_bottom = self.fiterPreProcessing(img, x_left, y_top, x_right, y_bottom)

        #フィルタ結果を画像に適応
        mosaic_img = img
        size = (x_right-x_left, y_bottom-y_top)
        face_img = cv2.resize(face_img, None, fx = self.ratio, fy = self.ratio, interpolation=cv2.INTER_NEAREST)
        face_img = cv2.resize(face_img, size, interpolation=cv2.INTER_NEAREST)
        mosaic_img[y_top:y_bottom, x_left:x_right] = face_img
        return mosaic_img


    def blurFilter(self, img, x_left: int, y_top: int, x_right: int, y_bottom: int):
        """
        ぼかしフィルタ
        """
        face_img, x_left, y_top, x_right, y_bottom = self.fiterPreProcessing(img, x_left, y_top, x_right, y_bottom)
        
        #フィルタ結果を画像に適応
        blur_img = img
        size = (x_right-x_left, y_bottom-y_top)
        face_img = cv2.blur(face_img, (self.kernel_size, self.kernel_size))
        blur_img[y_top:y_bottom, x_left:x_right] = face_img
        return blur_img


    def applyFilter(self, img, filter_type:str = "mosaic"):
        """
        フィルタを適応した画像を取得
        
        Parameters
        ----------
        img : フィルタを適応させたい画像
        filter_type (string): フィルタの種類（mosaic, blur）
        """
        self.img_idx += 1
        filter = self.mosaicFilter
        if filter_type == "blur":
            filter = self.blurFilter
        
        filter_img = img.copy()

        #顔の領域を取得    
        filter_areas = self.fd.detectFaceAreas(img)
        if isinstance(filter_areas, list) == True:
            if len(filter_areas) == 0:
                return filter_img
    
        for i in range(len(filter_areas)):
            x_left = filter_areas[i][0]
            x_right = x_left + filter_areas[i][2]
            y_top = filter_areas[i][1]
            y_bottom = y_top + filter_areas[i][3]
            filter_img = filter(filter_img, x_left, y_top, x_right, y_bottom)
        
        return filter_img