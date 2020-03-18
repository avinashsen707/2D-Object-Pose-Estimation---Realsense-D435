import numpy as np
import cv2 
import pyrealsense2 as rs


def mouse_callback(event,x,y,flags,param):
      if event == cv2.EVENT_LBUTTONDOWN:
         print x, y


# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

#Start pipeline
profile = pipeline.start(config)




while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    colour_frame = frames.get_color_frame()

    color_image = np.asanyarray(colour_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())   

    cv2.imshow('RealSense', color_image)
    cv2.setMouseCallback('RealSense',mouse_callback) #Mouse callback
    
    #cv.imshow("Depth", depth_colormap)
    #cv.imshow('Mask', fgmask)
    if cv2.waitKey(25) == ord('q'):
        break
cv2.destroyAllWindows()
pipeline.stop()



