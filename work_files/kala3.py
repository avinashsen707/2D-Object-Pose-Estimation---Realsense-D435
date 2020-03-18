import random

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

        images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
        cv2.imshow("win", images)

        pts = pc.calculate(aligned_depth_frame)
        pc.map_to(aligned_color_frame)

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        key = cv2.waitKey(1)

        if key == ord("d"):
            min_distance = 1e-6
            v = pts.get_vertices()
            vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            h, w, _ = color_image.shape
            sdk_pts = []
            projected_pts = []
            counter = 0
            for x in range(h):
                for y in range(w):
                    z = aligned_depth_frame.get_distance(y, x)
                    if z:
                        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [y, x], z)
                        if (np.math.fabs(point[0]) >= min_distance
                                or np.math.fabs(point[1]) >= min_distance
                                or np.math.fabs(point[2]) >= min_distance):
                            projected_pts.append([round(point[0], 8), round(point[1], 8), point[2], 3])
                    else:
                        # Ignoring pixels which doesn't have depth value
                        # print("No info at:", [y, x], z)
                        counter += 1
                        pass  

            for i in range(pts.size()):
                if (np.math.fabs(vertices[i][0]) >= min_distance
                        or np.math.fabs(vertices[i][1]) >= min_distance
                        or np.math.fabs(vertices[i][2]) >= min_distance):
                    sdk_pts.append(vertices[i])
            
            # Checking if the number of points generated are the same
            assert len(projected_pts) == len(sdk_pts)  # PASS

            print("Number of pixels ignored:", counter)

            rnd = random.randint(0, len(projected_pts))
            print("Projected point:",
                  "index =", rnd,
                  "x =", projected_pts[rnd][0],
                  "y =", projected_pts[rnd][1],
                  "z =", projected_pts[rnd][2]
                  )
            print("SDK point:",
                  "index =", rnd,
                  "x =", sdk_pts[rnd][0],
                  "y =", sdk_pts[rnd][1],
                  "z =", sdk_pts[rnd][2]
                  )
        if key == ord("q"):
            break
finally:
    pipeline.stop()
