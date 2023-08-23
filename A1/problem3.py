import numpy as np
import matplotlib.pyplot as plt


# Plot 2D points
def displaypoints2d(points):
  plt.figure(0)
  plt.plot(points[0,:],points[1,:], '.b')
  plt.xlabel('Screen X')
  plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(points[0,:], points[1,:], points[2,:], 'b')
  ax.set_xlabel("World X")
  ax.set_ylabel("World Y")
  ax.set_zlabel("World Z")


def cart2hom(points):
  """ Transforms from cartesian to homogeneous coordinates.

  Args:
    points: a np array of points in cartesian coordinates

  Returns:
    points_hom: a np array of points in homogeneous coordinates
  """

  #
  # You code here
  #
  k = points.shape[1]
  a = np.ones(k)
  points_hom = np.row_stack((points, a))

  return np.array(points_hom)

def hom2cart(points):
  """ Transforms from homogeneous to cartesian coordinates.

  Args:
    points: a np array of points in homogenous coordinates

  Returns:
    points_hom: a np array of points in cartesian coordinates
  """

  #
  # You code here
  #
  points_cart=np.delete(points,points.shape[0]-1,axis=0)
  return np.array(points_cart)


def gettranslation(v):
  """ Returns translation matrix T in homogeneous coordinates for translation by v.

  Args:
    v: 3d translation vector

  Returns:
    T: translation matrix in homogeneous coordinates
  """

  #
  # You code here
  #
  T=([[1,0,0,v[0]],[0,1,0,v[1]],[0,0,1,v[2]],[0,0,0,1]])
  return np.array(T)
def getxrotation(d):
  """ Returns rotation matrix Rx in homogeneous coordinates for a rotation of d degrees around the x axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rx: rotation matrix
  """

  #
  # You code here
  #
  R_x=([[1,0,0,0],[0,np.cos(d*np.pi/180),-np.sin(d*np.pi/180),0],[0,np.sin(d*np.pi/180),np.cos(d*np.pi/180),0],[0,0,0,1]])
  return np.array(R_x)

def getyrotation(d):
  """ Returns rotation matrix Ry in homogeneous coordinates for a rotation of d degrees around the y axis.

  Args:
    d: degrees of the rotation

  Returns:
    Ry: rotation matrix
  """

  #
  # You code here
  #
  R_y=([[np.cos(d*np.pi/180),0,np.sin(d*np.pi/180),0],[0,1,0,0],[-np.sin(d*np.pi/180),0,np.cos(d*np.pi/180),0],[0,0,0,1]])
  return np.array(R_y)

def getzrotation(d):
  """ Returns rotation matrix Rz in homogeneous coordinates for a rotation of d degrees around the z axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rz: rotation matrix
  """

  #
  # You code here
  #
  R_z=([[np.cos(d*np.pi/180),-np.sin(d*np.pi/180),0,0],[np.sin(d*np.pi/180),np.cos(d*np.pi/180),0,0],[0,0,1,0],[0,0,0,1]])
  return np.array(R_z)


def getcentralprojection(principal, focal):
  """ Returns the (3 x 4) matrix L that projects homogeneous camera coordinates on homogeneous
  image coordinates depending on the principal point and focal length.
  
  Args:
    principal: the principal point, 2d vector
    focal: focal length

  Returns:
    L: central projection matrix
  """

  #
  # You code here
  #
  L=np.array([[focal,0,principal[0],0],[0,focal,principal[1],0],[0,0,1,0]])
  return np.array(L)

def getfullprojection(T, Rx, Ry, Rz, L):
  """ Returns full projection matrix P and full extrinsic transformation matrix M.

  Args:
    T: translation matrix
    Rx: rotation matrix for rotation around the x-axis
    Ry: rotation matrix for rotation around the y-axis
    Rz: rotation matrix for rotation around the z-axis
    L: central projection matrix

  Returns:
    P: projection matrix
    M: matrix that summarizes extrinsic transformations
  """

  #
  # You code here
  #
  M=Rz.dot(Rx.dot(Ry.dot(T)))
  P=L.dot(M)
  return np.array(P),np.array(M)


def projectpoints(P, X):
  """ Apply full projection matrix P to 3D points X in cartesian coordinates.

  Args:
    P: projection matrix
    X: 3d points in cartesian coordinates

  Returns:
    x: 2d points in cartesian coordinates
  """

  #
  # You code here
  #
  x=cart2hom(X)
  x=P.dot(x)
  x=hom2cart(x)
  return np.array(x)

def loadpoints():
  """ Load 2D points from obj2d.npy.

  Returns:
    x: np array of points loaded from obj2d.npy
  """

  #
  # You code here
  #
  points =np.load('data/obj2d.npy')
  return points

def loadz():
  """ Load z-coordinates from zs.npy.

  Returns:
    z: np array containing the z-coordinates
  """

  #
  # You code here
  #
  z=np.load('data/zs.npy')
  return np.array(z)

def invertprojection(L, P2d, z):
  """
  Invert just the projection L of cartesian image coordinates P2d with z-coordinates z.

  Args:
    L: central projection matrix
    P2d: 2d image coordinates of the projected points
    z: z-components of the homogeneous image coordinates

  Returns:
    P3d: 3d cartesian camera coordinates of the points
  """

  #
  # You code here
  #

  inv_L=np.linalg.pinv(L)
  points_hom = cart2hom(P2d)
  k=P2d.shape[1]
  for i in range (k):
      points_hom[:,i] = points_hom[:,i]*z[:,i]
  P3d = inv_L.dot(points_hom)
  P3d = np.delete(P3d,P3d.shape[0]-1,axis=0)
  return np.array(P3d)


def inverttransformation(M, P3d):
  """ Invert just the model transformation in homogeneous coordinates
  for the 3D points P3d in cartesian coordinates.

  Args:
    M: matrix summarizing the extrinsic transformations
    P3d: 3d points in cartesian coordinates

  Returns:
    X: 3d points after the extrinsic transformations have been reverted
  """

  #
  # You code here
  #
  P3d_hom = cart2hom (P3d)
  inv_M=np.linalg.inv(M)
  P3d=inv_M.dot(P3d_hom)
  return P3d

def p3multiplecoice():
  '''
  Change the order of the transformations (translation and rotation).
  Check if they are commutative. Make a comment in your code.
  Return 0, 1 or 2:
  0: The transformations do not commute.
  1: Only rotations commute with each other.
  2: All transformations commute.
  '''

  return -1



t = np.array([-27.1, -2.9, -3.2])
principal_point = np.array([8, -10])
focal_length = 8

# model transformations
T = gettranslation(t)
Ry = getyrotation(135)
Rx = getxrotation(-30)
Rz = getzrotation(90)
print(T)
print(Ry)
print(Rx)
print(Rz)

K = getcentralprojection(principal_point, focal_length)

P,M = getfullprojection(T, Rx, Ry, Rz, K)
print(P)
print(M)

points = loadpoints()
displaypoints2d(points)

z = loadz()
Xt = invertprojection(K, points, z)

Xh = inverttransformation(M, Xt)

worldpoints = hom2cart(Xh)
displaypoints3d(worldpoints)

points2 = projectpoints(P, worldpoints)
displaypoints2d(points2)

plt.show()
