import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from PIL import Image
from problem1 import Problem1
def problem1():

    def show_matches(im1, im2, pairs):
        plt.figure()
        plt.title("Keypoint matches")
        plt.imshow(np.append(im1, im2, axis=1), "gray", interpolation=None)
        plt.axis("off")
        shift = im1.shape[1]
        colors = pl.cm.viridis( np.linspace(0, 1 , pairs.shape[0]))
        for i in range(pairs.shape[0]):
            plt.scatter(x=pairs[i,0], y=pairs[i,1], color=colors[i])
            plt.scatter(x=pairs[i,2]+shift, y=pairs[i,3], color=colors[i])

    def show_image(im, title=""):
        plt.figure()
        plt.title(title)
        plt.imshow(im, "gray", interpolation=None)
        plt.axis("off")

    def stitch_images(im1, im2, H):
        h, w = im1.shape
        warped = np.zeros((h, 2*w))
        warped[:,:w] = im1
        im2 = Image.fromarray(im2)
        im3 = im2.transform(size=(2*w, h),
                                    method=Image.PERSPECTIVE,
                                    data=H.ravel(),
                                    resample=Image.BICUBIC)
        im3 = np.array(im3)
        warped[im3 > 0] = im3[im3 > 0]
        return warped

    # RANSAC Parameters
    ransac_threshold = 5.0  # inlier threshold
    p = 0.35                # probability that any given correspondence is valid
    k = 4                   # number of samples drawn per iteration
    z = 0.99                # total probability of success after all iterations

    P1 = Problem1()

    # load images
    im1 = plt.imread("data/a4p1a.png")
    im2 = plt.imread("data/a4p1b.png")

    # load keypoints
    data = np.load("data/keypoints.npz")
    keypoints1 = data['keypoints1']
    keypoints2 = data['keypoints2']

    # load SIFT features for the keypoints
    data = np.load("data/features.npz")
    features1 = data['features1']
    features2 = data['features2']

    # find matching keypoints
    distances = P1.euclidean_square_dist(features1, features2)
    pairs = P1.find_matches(keypoints1, keypoints2, distances)
    show_matches(im1, im2, pairs)

    # Compute homography matrix via ransac
    n_iters = P1.ransac_iters(p, k, z)
    H, num_inliers, inliers = P1.ransac(pairs, n_iters, k, ransac_threshold)
    print('Number of inliers:', num_inliers)
    warped = stitch_images(im1, im2, H)
    show_image(warped, "Ransac Homography")

    # recompute homography matrix based on inliers
    H = P1.recompute_homography(inliers)
    warped = stitch_images(im1, im2, H)
    show_image(warped, "Recomputed Homography")
    plt.show()


if __name__ == "__main__":
    problem1()
