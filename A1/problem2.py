import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    #
    plt.imshow(img)

def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    #
    # You code here
    #
    data=np.load("data/bayerdata.npy")
    return np.array(data)

def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    #
    # You code here
    #
    global H,W
    H,W = bayerdata.shape
    r = np.zeros((H, W))
    g = np.zeros((H, W))
    b = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if ((i+j)%2 == 0):
                g[i][j]=bayerdata[i][j]
            elif ((i+j)%2!=0 and i%2 ==0):
                b[i][j]=bayerdata[i][j]
            else:
                r[i][j]=bayerdata[i][j]
    return np.array(r),np.array(g),np.array(b)


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    global img
    img=np.zeros((H,W,3))
    img[:,:,0]=r[:,:]
    img[:,:,1]=g[:,:]
    img[:,:,2]=b[:,:]
    return np.array(img)

def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    r_weight=np.array([[0.25,0,0.25],[0.5,1,0.5],[0.25,0,0.25]])
    g_weight=np.array([[0,0.25,0],[0.25,1,0.25],[0,0.25,0]])
    b_weight=np.array([[0.25,0.5,0.25],[0,1,0],[0.25,0.5,0.25]])
    r_interpolation = convolve(r, r_weight,mode='nearest')
    g_interpolation = convolve(g, g_weight,mode='nearest')
    b_interpolation = convolve(b, b_weight,mode='nearest')
    img = np.zeros((H,W,3))
    img[:,:,0]=r_interpolation[:,:]
    img[:,:,1]=g_interpolation[:,:]
    img[:,:,2]=b_interpolation[:,:]
    return np.array(img)

data = loaddata("data/bayerdata.npy")
r, g, b = separatechannels(data)
plt.figure('img')
img = assembleimage(r, g, b)
display_image(img)
plt.figure('img_interpolated')
img_interpolated = interpolate(r, g, b)
display_image(img_interpolated)
plt.show()