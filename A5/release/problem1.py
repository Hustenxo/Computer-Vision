from functools import partial
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.ndimage import convolve


####################
# Provided functions
####################


conv2d = partial(convolve, mode="mirror")


def gauss2d(fsize, sigma):
    """ Create a 2D Gaussian filter

    Args:
        fsize: (w, h) dimensions of the filter
        sigma: width of the Gaussian filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def downsample(img, fsize=(5, 5), sigma=1.4):
    """
    Downsampling an image by a factor of 2

    Args:
        img: image as (h, w) np.array
        fsize and sigma: parameters for Gaussian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as (h/2, w/2) np.array
    """
    g_k = gauss2d(fsize, sigma)
    img = conv2d(img, g_k)
    return img[::2, ::2]


def gaussian_pyramid(img, nlevels=3, fsize=(5, 5), sigma=1.4):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        fsize: gaussian kernel size
        sigma: sigma of gaussian kernel

    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    pyramid = [img]
    for i in range(0, nlevels - 1):
        pyramid.append(downsample(pyramid[i], fsize, sigma))

    return pyramid


def resize(arr, shape):
    """ Resize an image to target shape

    Args:
        arr: image as (h, w) np.array
        shape: target size (h', w') as tuple

    Returns:
        resized image as (h', w') np.array
    """
    return np.array(Image.fromarray(arr).resize(shape[::-1]))


######################
# Basic Lucas-Kanade #
######################


def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives

    Args:
        im1: first image as (h, w) np.array
        im2: second image as (h, w) np.array

    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
                    as (h, w) np.array
    """
    #
    # You code here
    #
    sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])# horizont Sobel
    sobel_x = sobel_x / np.sum(abs(sobel_x))
    sobel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])# vertical Sobel
    sobel_y = sobel_y / np.sum(abs(sobel_y))
    Ix = conv2d(im1, sobel_x)
    Iy = conv2d(im1, sobel_y)
    It = np.array(im2 - im1)

    return np.array(Ix), np.array(Iy), np.array(It)



def compute_motion(Ix, Iy, It, patch_size=40):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (h, w) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (h, w) np.array
        v: optical flow in y direction as (h, w) np.array
    """
    #
    # You code here
    #
    R = np.ones((patch_size, patch_size))
    I_x2 = conv2d(Ix **2 , R)
    I_xy = conv2d(Ix * Iy , R)
    I_y2 = conv2d(Iy **2 , R)
    h=Ix.shape[0]
    w=Ix.shape[1]
    u = np.empty((h, w))
    v = np.empty((h, w))
    I_xt = conv2d(Ix * It, R)
    I_yt = conv2d(Iy * It, R)
    # LK in center of region
    for i in range(h):
        for j in range(w):
            structure_tensor = np.array([[I_x2[i, j], I_xy[i, j]],[I_xy[i, j], I_y2[i, j]]])
            time = -np.array([[I_xt[i, j]], [I_yt[i, j]]])
            flow = np.linalg.inv(structure_tensor) @ time
            u[i, j] = flow[0]
            v[i, j] = flow[1]

    return np.array(u), np.array(v)

def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image as (h, w) np.array
        u, v: optical flow in x and y direction each as (h, w) np.array

    Returns:
        im_warp: warped image as (h, w) np.array
    """
    #
    # You code here
    #
    h = im.shape[0]
    w = im.shape[1]
    h_1, w_1 = np.mgrid[0:h, 0:w].astype(np.float)
    mesh_h_warp = u + h_1    # --> horizontal
    mesh_w_warp = v + w_1    # --> vertical
    values = im.flatten() #(h, w) np.array-->(n, )np.array
    points = np.stack([mesh_h_warp.flatten(),mesh_w_warp.flatten()], axis=-1)
    im_warp = interpolate.griddata(points, values, (h_1, w_1), method='nearest')

    return np.array(im_warp)

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade.
    Args:
        im1, im2: Images as (h, w) np.array
    
    Returns:
        Cost as float scalar
    """
    #
    # You code here
    #
    Cost = (im1 - im2)**2
    return np.sum(Cost)

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(pyramid1, pyramid2, n_iter=10):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.

    Args:
        pyramid1, pyramid2: Gaussian pyramids corresponding to
                            im1 and im2, in fine to coarse order
        n_iter: number of refinement iterations

    Returns:
        u: OF in x direction as np.array
        v: OF in y direction as np.array
    """
    #
    # You code here
    #
    u = v = np.zeros(pyramid1[-1].shape)
    for i in reversed(range(len(pyramid1))):
        x = pyramid1[i].shape
        v =u = resize(u, x) * 2
        for j in range(n_iter):
            wrap = warp(pyramid1[i], u, v)
            Ix, Iy, It = compute_derivatives(wrap, pyramid2[i])
            delta_u, delta_v = compute_motion(Ix, Iy, It)
            delta_u = np.nan_to_num(delta_u)
            delta_v = np.nan_to_num(delta_v)
            u = delta_u + u
            v = delta_v + v
    return np.array(u), np.array(v)
