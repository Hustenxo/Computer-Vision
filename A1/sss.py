import numpy as np
import matplotlib.pyplot as plt

def displaypoints2d(points):
  plt.figure(0)
  plt.plot(points[0,:],points[1,:], '.b')
  plt.xlabel('Screen X')
  plt.ylabel('Screen Y')
  plt.show()

points=np.load('data/obj2d.npy')
print(points)
displaypoints2d(points)
