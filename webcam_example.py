import cv2
from facefilter.face_filter import FaceFilter

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    ff = FaceFilter()

    while(True):
        ret, frame = capture.read()
        windowsize = (1280, 720)
        frame = cv2.resize(frame, windowsize)
        filter_img = ff.applyFilter(frame, filter_type="blur")
        cv2.imshow('title',filter_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()