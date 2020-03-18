import cv2
import numpy as np



cv2.namedWindow('mouseRGB')

param1 =10
capture = cv2.VideoCapture(0)

while(True):

    ret, frame = capture.read()

    cv2.imshow('mouseRGB', frame)
    
    
    
    def mouseRGB(event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
			colorsB = frame[y,x,0]
			colorsG = frame[y,x,1]
			colorsR = frame[y,x,2]
			colors = frame[y,x]
			print(param1)
    cv2.setMouseCallback('mouseRGB',mouseRGB)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
