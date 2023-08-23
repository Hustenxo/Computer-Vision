import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import flow_to_color
import problem1 as p1

#
# Problem 1
#

def problem1():
    def show_two(gray_im, flow_im):
        fig = plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
        (ax1, ax2) = fig.subplots(1, 2)
        ax1.imshow(gray_im, "gray", interpolation=None)
        ax2.imshow(flow_im)
        plt.show()

    # Loading the image and scaling to [0, 1]
    im1 = np.array(Image.open("data/a5p1a.png")) / 255.0
    im2 = np.array(Image.open("data/a5p1b.png")) / 255.0
    #
    # Basic implementation
    #
    Ix, Iy, It = p1.compute_derivatives(im1, im2) # gradients
    u, v = p1.compute_motion(Ix, Iy, It) # flow

    # stacking for visualization
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)

    # warping 1st image to the second
    im1_warped = p1.warp(im1, u, v)
    cost = p1.compute_cost(im1_warped, im2)
    print(f"Cost (basic): {cost:4.3e}")

    #
    # Iterative coarse-to-fine implementation
    #
    n_iter = 4 # number of iterations
    n_levels = 3 # levels in Gaussian pyramid

    pyr1 = p1.gaussian_pyramid(im1, nlevels=n_levels)
    pyr2 = p1.gaussian_pyramid(im2, nlevels=n_levels)

    u, v = p1.coarse_to_fine(pyr1, pyr2, n_iter)

    # warping 1st image to the second
    im1_warped = p1.warp(im1, u, v)
    cost = p1.compute_cost(im1_warped, im2)
    print(f"Cost (coarse-to-fine): {cost:4.3e}")

    # stacking for visualization
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)


if __name__ == "__main__":
    problem1()
