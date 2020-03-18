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
		cv2.imshow("mouseRGB", images)
		pts = pc.calculate(aligned_depth_frame)
		pc.map_to(aligned_color_frame)
		key = cv2.waitKey(1)
		min_distance = 1e-6
		v = pts.get_vertices()
		vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
		h, w, _ = color_image.shape
		projected_pts = []
		def mouseRGB(event,x,y,flags,param):
			if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
				z = aligned_depth_frame.get_distance(y, x)
				point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [y, x], z)				
				print("Projected point:", "x =", point[0],"y =", point[1],"z =", point[2])
		cv2.setMouseCallback('mouseRGB',mouseRGB)
		if key == ord("q"):
			break
finally:
    pipeline.stop()
