import cv2
import numpy as np
import pyrealsense2 as rs
import time
from matplotlib import pyplot as plt

MIN_MATCH_COUNT=50
ANPLOT = 1
flag=1
ENABLE_FILTER =1

	
def update_line(hl, xx,yy,zz):
	xdata, ydata, zdata = hl._verts3d
	hl.set_xdata(xx)
	hl.set_ydata(yy)
	hl.set_3d_properties(zz)
		
	plt.draw()	
	
def Estimate(coX, coY):
	measured = np.array([[np.float32(coX)], [np.float32(coY)]])
	kf.correct(measured)
	predicted = kf.predict()
	return predicted

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)
profile = pipeline.get_active_profile()
depth_profile =rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
pc = rs.pointcloud()
colorizer = rs.colorizer()
align_to = rs.stream.color
align = rs.align(align_to)
if ENABLE_FILTER:
	#dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
	spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
	temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
	filter_smooth_alpha = rs.option.filter_smooth_alpha
	filter_smooth_delta = rs.option.filter_smooth_delta
	#filter_holes_fill = rs.option.hole_filling_filter
	temp_filter.set_option(filter_smooth_alpha, 0.05)
	temp_filter.set_option(filter_smooth_delta, 80)
	#temp_filter.set_option(filter_holes_fill,5)
	

	

if ANPLOT:
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import pyplot as plt
	map = plt.figure()
	map_ax = Axes3D(map)
	map_ax.autoscale(enable=True, axis='both', tight=True)
	map_ax.set_xlim3d([-.3, .3])
	map_ax.set_ylim3d([-.3, .3])
	map_ax.set_zlim3d([-.8, .8 ])
	hl, = map_ax.plot3D([0], [0], [0])
	#hl, = map_ax.plot3D([0], [0], [0],marker='o',color='r', linestyle='None', markersize = 10.0)
	

try:
	while flag==1:
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)
		aligned_depth_frame = aligned_frames.get_depth_frame()	
		#if ENABLE_FILTER:
			
			#filtered = temp_filter.process(aligned_depth_frame)
			#filtered = dec_filter.process(filtered)
			#filtered = spat_filter.process(filtered)
			
		aligned_color_frame = aligned_frames.get_color_frame()
		depth_colormap =np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
		color_image = np.asanyarray(aligned_color_frame.get_data())
		images = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		cv2.imshow("mouseRGB", depth_colormap)
		pts = pc.calculate(aligned_depth_frame)
		pc.map_to(aligned_color_frame)
		min_distance = 1e-6
		v = pts.get_vertices()
		vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
		projected_pts = []
		key = cv2.waitKey(1)
		# ___________________________________________________________________________....
		detector = cv2.xfeatures2d.SIFT_create()
		FLANN_INDEX_KDITREE = 0
		flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
		flann = cv2.FlannBasedMatcher(flannParam, {})
		trainImg = cv2.imread('TrainingData/tr_data.png', 0)
	
		trainImg = cv2.imread('TrainingData/tr_data.png')
		trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
		template = cv2.imread('TrainingData/tr_data.png',0)
		w, h = template.shape[::-1]

		res = cv2.matchTemplate(trainImg_gray,template,cv2.TM_CCOEFF_NORMED)
		threshold = 0.8
		loc = np.where( res >= threshold)
		for pt in zip(*loc[::-1]):
			cv2.rectangle(trainImg, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		cv2.imwrite('res.png',trainImg)
		
		(trainKP, trainDesc) = detector.detectAndCompute(trainImg, None)
		#print trainDesc
		QueryImg = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
		(queryKP, queryDesc) = detector.detectAndCompute(QueryImg, None)
		matches = flann.knnMatch(queryDesc, trainDesc, k=2)
		goodMatch = []
		for (m, n) in matches:
			if m.distance < 0.75 * n.distance:
				goodMatch.append(m)
		if len(goodMatch) > MIN_MATCH_COUNT:
			tp = []
			qp = []
			for m in goodMatch:
				tp.append(trainKP[m.trainIdx].pt)
				qp.append(queryKP[m.queryIdx].pt)
			(tp, qp) = np.float32((tp, qp))
			#print tp,qp
			(H, status) = cv2.findHomography(tp, qp, cv2.RANSAC, 2.0)
			(h, w) = trainImg.shape
			h = h-20
			w= w-20
			trainBorder = np.float32([[[100, 100], [100, h - 100], [w - 100, h- 100], [w - 100, 100]]])
			queryBorder = cv2.perspectiveTransform(trainBorder, H)
			print ('border',np.int32(queryBorder))
			tem = np.int32(queryBorder).reshape(4,2)	
			#print tem		
			if ANPLOT:
				#print ('tem is ',tem)
				x = tem[:,0]
				#print ('x is ',x)
				y = tem[:,1]
				z = np.array(y,dtype='float32')
				for i in range(4):
					 
					if (i==0 or i==1):
						xoff=20
					else:
						xoff=-20
					if (i==0 or i==3):
						yoff=20
					else: 
						yoff=-20
					val = aligned_depth_frame.get_distance(x[i]+xoff, y[i]+yoff)
					if val:
						z[i] = val
					else:
						z[i] = np.nan						
				z = np.append(z,z[0])
				x = np.append(x,x[0])
				y = np.append(y,y[0])
				xx = np.array(x,dtype='float32')
				yy = np.array(y,dtype='float32')
				zz = z
				#print xx,yy,zz
				print 'hi'
				for i in range(4):
					point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x[i] ,y[i]], z[i])	

					#print point				
					xx[i] = point[0]
					yy[i] = point[1]
					zz[i] = point[2]
				xx[4] = xx[0]
				yy[4] = yy[0]
				zz[4] = zz[0]
				#print('x is =',xx[1],'Y is=',yy[1],'Z is =',zz[1])
					
				update_line(hl, xx,yy,zz)
				#map_ax.plot3D([np.mean(xx),np.mean(xx)], [np.mean(yy),np.mean(yy)], [np.mean(zz),np.mean(zz)],marker='o',color='r', linestyle='None', markersize = 10.0)
				plt.show(block=False)
				plt.pause(0.01)
				
			cv2.polylines(images, [np.int32(queryBorder)], True, (0, 255, 0), 5)
			time.sleep(0.1)
		else:
			print 'Not Enough match found- %d/%d' % (len(goodMatch),MIN_MATCH_COUNT)
		cv2.imshow('result', images)
		if key == ord('q'):
			break
finally:
	pipeline.stop()


