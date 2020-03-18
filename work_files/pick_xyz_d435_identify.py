import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)

profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

pc = rs.pointcloud()
colorizer = rs.colorizer()

align_to = rs.stream.color
align = rs.align(align_to)
try:
	while True:
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)
		aligned_depth_frame = aligned_frames.get_depth_frame()
		aligned_color_frame = aligned_frames.get_color_frame()
		depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
		color_image = np.asanyarray(aligned_color_frame.get_data())
		images = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)		
		#cv2.imshow("mouseRGB", images)
		pts = pc.calculate(aligned_depth_frame)
		pc.map_to(aligned_color_frame)
		
		min_distance = 1e-6
		v = pts.get_vertices()
		vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz		
		projected_pts = []
		#___________________________________________________________________________	
		
		detector=cv2.xfeatures2d.SIFT_create()
		FLANN_INDEX_KDITREE=0
		flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
		flann=cv2.FlannBasedMatcher(flannParam,{})
		trainImg=cv2.imread("TrainingData/tr_data.png",0)
		trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
		QueryImg=cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
		queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
		matches=flann.knnMatch(queryDesc,trainDesc,k=2)
		goodMatch=[]
		for m,n in matches:
			if(m.distance<0.75*n.distance):
				goodMatch.append(m)
		if(len(goodMatch)>MIN_MATCH_COUNT):
			tp=[]
			qp=[]
			for m in goodMatch:
				tp.append(trainKP[m.trainIdx].pt)
				qp.append(queryKP[m.queryIdx].pt)
			tp,qp=np.float32((tp,qp))
			H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
			h,w=trainImg.shape
			trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
			queryBorder=cv2.perspectiveTransform(trainBorder,H)
			#print queryBorder
			cv2.polylines(images,[np.int32(queryBorder)],True,(0,255,0),5)
			time.sleep(1)
		else:
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
        cv2.imshow('result',images)
    	if key == ord("q"):
			break
finally:
	pipeline.stop()
