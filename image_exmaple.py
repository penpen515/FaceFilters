import cv2
from facefilter.face_filter import FaceFilter

if __name__ == '__main__':
    filename = "Please input filename here"
    img = cv2.imread(filename)
    
    ff = FaceFilter()
    
    #mosaic
    mosaic_image = ff.applyFilter(img)
    #blur
    blur_image = ff.applyFilter(img, filter_type="blur")
    cv2.imshow('mosaic',mosaic_image)
    cv2.imshow('blur',blur_image)
    cv2.imshow("originai", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()