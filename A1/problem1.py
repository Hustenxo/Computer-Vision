import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
def load_image(path):
    return plt.imread(path)
def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    #
    plt.imshow(img)

def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    #
    img = np.load(path)
    return np.array(img)
def mirror_horizontal(img1):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    #
    # You code here
    #
    img = img1[::-1,...]
    return img
def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    #
    plt.figure('f2')
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)

img = load_image("data/a1p1.png")
plt.figure('f1')
plt.subplot(1, 3, 1)
plt.title('img')
display_image(img)
save_as_npy("a1p1.npy", img)
img1 = load_npy("a1p1.npy")
plt.subplot(1, 3, 2)
plt.title('img1')
display_image(img1)
img2 = mirror_horizontal(img1)
plt.subplot(1, 3, 3)
plt.title('img2')
display_image(img2)
plt.title('img1 and 2')
display_images(img1, img2)
plt.show()

