import numpy as np

def surface_normal_newell(poly):

    n = np.array([0.0, 0.0, 0.0])

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly),:]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2]) 
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm

    return normalised


def surface_normal_cross(poly):

    n = np.cross(poly[1,:]-poly[0,:],poly[2,:]-poly[0,:])

    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm

    return normalised


def test_surface_normal3():
    """ should return:

        Newell:
        Traceback (most recent call last):
          File "demo_newells_surface_normals.py", line 96, in <module>
            test_surface_normal3()
          File "demo_newells_surface_normals.py", line 58, in test_surface_normal3
            print "Newell:", surface_normal_newell(poly) 
          File "demo_newells_surface_normals.py", line 24, in surface_normal_newell
            raise ValueError('zero norm')
        ValueError: zero norm
    """
    poly = np.array([[1.0,0.0,0.0],
                     [1.0,0.0,0.0],
                     [1.0,0.0,0.0]])
    print "Newell:", surface_normal_newell(poly) 


def test_surface_normal2():
    """ should return:

        Newell: [ 0.08466675 -0.97366764 -0.21166688]
        Cross : [ 0.08466675 -0.97366764 -0.21166688]
    """
    poly = np.array([[6.0,1.0,4.0],
                     [7.0,0.0,9.0],
                     [1.0,1.0,2.0]])
    print "Newell:", surface_normal_newell(poly)
    print "Cross :", surface_normal_cross(poly)


def test_surface_normal1():
    """ should return:

        Newell: [ 0.  0. -1.]
        Cross : [ 0.  0. -1.]
    """
    poly = np.array([[0.0,1.0,0.0],
                     [1.0,1.0,0.0],
                     [1.0,0.0,0.0]])
    print "Newell:", surface_normal_newell(poly) 
    print "Cross :", surface_normal_cross(poly)


print "Test 1:"
test_surface_normal1()
print "\n"

print "Test 2:"
test_surface_normal2()
print "\n"

print "Test 3:"
test_surface_normal3()
