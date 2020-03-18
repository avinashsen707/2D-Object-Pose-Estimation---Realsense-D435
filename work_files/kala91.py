import numpy as np

# coordinates (XYZ) of C1, C2, C4 and C5
XYZ = np.array([
        [0.274791784, -1.001679346, -1.851320839, 0.365840754],
        [-1.155674199, -1.215133985, 0.053119249, 1.162878076],
        [1.216239624, 0.764265677, 0.956099579, 1.198231236]])

# Inital guess of the plane
p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]

def f_min(X,p):
    plane_xyz = p[0:3]
    distance = (plane_xyz*X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, signal, X):
    return f_min(X, params)

from scipy.optimize import leastsq
sol = leastsq(residuals, p0, args=(None, XYZ))[0]

print("Solution: ", sol)
print("Old Error: ", (f_min(XYZ, p0)**2).sum())
print("New Error: ", (f_min(XYZ, sol)**2).sum())
