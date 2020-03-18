

import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
 
 
def update_line(hl, new_data):
	xdata, ydata, zdata = hl._verts3d
	hl.set_xdata(new_data[0:2])
	hl.set_ydata(new_data[2:4])
	hl.set_3d_properties(new_data[4:6])
	plt.draw()
 
 
map = plt.figure()
map_ax = Axes3D(map)
map_ax.autoscale(enable=True, axis='both', tight=True)
 
# # # Setting the axes properties
map_ax.set_xlim3d([0.0, 10.0])
map_ax.set_ylim3d([0.0, 10.0])
map_ax.set_zlim3d([0.0, 10.0])
 
hl, = map_ax.plot3D([0], [0], [0])
 
update_line(hl, np.int32((1,2, 2,4,3,4)))
plt.show(block=False)
plt.pause(1)
 
update_line(hl, np.int32((2,3, 1,4,2,2,4)))
plt.show(block=False)
plt.pause(1)
 
update_line(hl, np.int32((3,2, 3,4,3,2)))
plt.show(block=True)

'''
def update_line(hl, new_data):
	xdata, ydata, zdata = hl._verts3d
	hl.set_xdata(list(np.append(xdata, new_data[0])))
	hl.set_ydata(list(np.append(ydata, new_data[1])))
	hl.set_3d_properties(list(np.append(zdata, new_data[2])))
	plt.draw()
	'''
