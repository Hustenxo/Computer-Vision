import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from PIL import Image
class Problem1:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """

        #
        # You code here
        #
        distances = np.ones((features2.shape[1], features1.shape[1]))
        for i in range(features2.shape[1]):
            for j in range(features1.shape[1]):
                distances[i, j] = np.linalg.norm(features2[:, i] - features1[:, j])

        return np.array(distances)


    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """

        #
        # You code here
        #
        n = distances.shape[0]
        m = distances.shape[1]
        pairs = np.zeros((min(n, m), 4))

        if n < m:#Image reconstruction based on large images
            for i in range(n):
                np.where(distances[i, :] == min(distances[i, :]))
                pairs[i, :2] = p1[np.where(distances[i, :] == min(distances[i, :]))]
                pairs[i, 2:] = p2[i, :]
        else:
            for i in range(m):
                pairs[i, :2] = p1[i, :]
                pairs[i, 2:] = p2[np.where(distances[:, i] == min(distances[:, i]))]

        return np.array(pairs)


    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """

    #
    # You code here
    #
        n = p1.shape[0]
        m = p2.shape[0]
        if m >= n:
            x = np.random.randint(0, n, size=k)

        else:
            x = np.random.randint(0, m, size=k)
        sample1 = p1[x, :]
        sample2 = p2[x, :]
        return np.array(sample1), np.array(sample2)


    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormalized cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """
    #
    # You code here
    # pdf I8 page 23
        s = (1 / 2) * np.max(np.abs(points), axis=0)
        t = np.mean(points, axis=0)
        T = np.array([[1 / s[0], 0, -t[0] / s[0]], [0, 1 / s[1], -t[1] / s[1]], [0, 0, 1]])
        x = np.ones(points.shape[0])
        x = x.reshape(-1,1)
        points = np.hstack((points, x))
        ps = (T @ points.T).T

        return np.array(ps), np.array(T)

    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices should be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2

        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

    #
    # You code here
    #
        A = np.empty((2 * p1.shape[0], 9))

        for i in range(p1.shape[0]):
            x_ = p2[i, 0]
            y_ = p2[i, 1]
            A[2 * i, :] = np.array(
                [0, 0, 0, p1[i, 0], p1[i, 1], p1[i, 2], -p1[i, 0] * y_, -p1[i, 1] * y_, -p1[i, 2] * y_])
            A[2 * i + 1, :] = np.array(
                [-p1[i, 0], -p1[i, 1], -p1[i, 2], 0, 0, 0, p1[i, 0] * x_, p1[i, 1] * x_, p1[i, 2] * x_])

        _, _, vh = np.linalg.svd(A)

        HC = vh[-1, :].reshape(3, -1)
        H = np.linalg.pinv(T2) @ HC @ T1
        # normalize
        if HC[-1,-1] != 0:
            HC = HC / HC[-1, -1]
        if H[-1, -1] != 0:
            H = H / H[-1, -1]

        return H, HC


    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix

        Returns:
            points: (l, 2) numpy array, transformed points
        """

    #
    # You code here
    #
        x = np.ones(p.shape[0])  # (1,l)
        x = x.reshape(-1, 1)  # (l,1)
        p_ = np.hstack((p, x))  # (l,3)

        points = (H @ p_.T).T  # (l,3)

        # normalization to reduce 3nd dimension in points (go back to non-homogenous coordinate)
        for i, point in enumerate(points):
            if point[-1] != 0:
                point = point / point[-1]
            else:
                point = np.zeros((1, 3))
            points[i, :] = point

        return np.array(points[:, :2])

    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2

        Returns:
            dist: (l, ) numpy array containing the distances
        """
        #
        # You code here
        #
        # d**2 = |H*p1-p2|**2+|x1-H**-1*x2|**2
        dist_1 = np.linalg.norm(self.transform_pts(p1, H) - p2, axis = 1 , keepdims = True) ** 2
        inv_H = np.linalg.pinv(H)
        dist_2 = np.linalg.norm(p1 - self.transform_pts(p2, inv_H), axis = 1 ,keepdims = True) ** 2
        dist = dist_1 + dist_2

        return np.array(dist)



    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance.

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold

        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
    #
    # You code here
    #

        if (dist <= threshold).any():
            N = np.sum(dist <= threshold)
            inliers = np.zeros((N, 4))
            x = np.where(dist <= threshold)
            for i in range (len(x)):
                inliers = pairs[x[i],:]

        else:
            N= 0
            inliers = np.zeros((N, 4))

        return N, inliers


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations

        Returns:
            minimum number of required iterations
        """

    #
    # You code here
    # pdf I8 page 28
        number = np.log(1 - z) / np.log(1 - p**k)
        return int(np.ceil(number)) #(int) wegen (TypeError: ‘numpy.float64‘ object cannot be interpreted as an integer)



    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold

        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """

    #
    # You code here
    #
        H = np.zeros((3, 3))
        max_inliers = 0

        for i in range(n_iters):
            p1, p2 = self.pick_samples(pairs[:, :2], pairs[:, 2:], k)
            p_p1, T_p1 = self.condition_points(p1)
            p_p2, T_p2 = self.condition_points(p2)
            H_1, _ = self.compute_homography(p_p1, p_p2, T_p1, T_p2)
            dist = self.compute_homography_distance(H_1, pairs[:, :2], pairs[:, 2:])
            N_1, inliers_1 = self.find_inliers(pairs, dist, threshold)

            if N_1 > max_inliers:
                H = H_1
                max_inliers = N_1
                inliers = inliers_1

        return np.array(H), max_inliers, np.array(inliers)


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points

        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
    #
    # You code here
    #

        if len(inliers) == 0:
            H = np.zeros((3, 3))
        else:
            p_p1, T_p1 = self.condition_points(inliers[:, :2])
            p_p2, T_p2 = self.condition_points(inliers[:, 2:])
            H, _ = self.compute_homography(p_p1, p_p2, T_p1, T_p2)

        return np.array(H)


