import numpy as np
from scipy.ndimage import convolve, maximum_filter
import cv2


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    #
    # You code here
    #



def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """
    #
    # You code here
    #


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    #
    # You code here
    #


def imagepatch_descriptors(gray, rows, cols):
    """ Get image patch descriptors for every interes point

        Args:
            img: (h, w) np.array with image gray values
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
        Returns:
            descriptors: (n, patch_size**2) np.array with image patch feature descriptors
    """

    #
    # You code here
    #


def match_interest_points(descriptors1, descriptors2):
    """ Brute-force match the interest points descriptors of two images using the cv2.BFMatcher function. 
    Select a reasonable distance measurement to be used and set "crossCheck=True".

    Args:
        descriptors1: (n, patch_size**2) np.array with image patch feature descriptors
        descriptors2: (n, patch_size**2) np.array with image patch feature descriptors
    Returns:
        matches: (m) list of matched descriptor pairs
    """
    #
    # You code here
    #



def load_img(path):
    color = Image.open(path)
    gray = color.convert("L")
    color = np.array(color) / 255
    gray = np.array(gray) / 255
    return color, gray

def show_points(img, rows, cols):
    plt.imshow(img, interpolation="none")
    plt.plot(cols, rows, "xr", linewidth=8)
    plt.axis("off")

def plot_heatmap(img, title=""):
    plt.imshow(img, "jet", interpolation="none")
    plt.axis("off")
    plt.title(title)

# Set paramters and load the image
sigma = 2
threshold = 3e-3
img_dir = os.path.join("data", "a3p2")
imgs_data = {}

for img_name in os.listdir(img_dir):
    color, gray = load_img(os.path.join(img_dir, img_name))

    # Generate filters and compute Hessian
    fx, fy = derivative_filters()
    gauss = gauss2d(sigma, (10, 10))
    I_xx, I_yy, I_xy = compute_hessian(gray, gauss, fx, fy)

    # Show components of Hessian matrix
    plt.figure()
    plt.subplot(1, 4, 1)
    plot_heatmap(I_xx, "I_xx")
    plt.subplot(1, 4, 2)
    plot_heatmap(I_yy, "I_yy")
    plt.subplot(1, 4, 3)
    plot_heatmap(I_xy, "I_xy")

    # Compute and show Hessian criterion
    criterion = compute_criterion(I_xx, I_yy, I_xy, sigma)
    plt.subplot(1, 4, 4)
    plot_heatmap(criterion, "Determinant of Hessian")

    # Show all interest points where criterion is greater than threshold
    rows, cols = np.nonzero(criterion > threshold)
    plt.figure()
    show_points(color, rows, cols)

    # Apply non-maximum suppression and show remaining interest points
    rows, cols = nonmaxsuppression(criterion, threshold)
    plt.figure()
    show_points(color, rows, cols)
    plt.show()

    # Get image patches around feature points as local descriptors
    descriptors = imagepatch_descriptors(gray, rows, cols)

    # Save computed interest points
    imgs_data[img_name] = [color, gray, rows, cols, descriptors]

# Load image data, interest points and descriptors
color1, _, rows1, cols1, descriptors1 = imgs_data["a3p2_0.png"]
color2, _, rows2, cols2, descriptors2 = imgs_data["a3p2_1.png"]
color3, _, rows3, cols3, descriptors3 = imgs_data["a3p2_2.png"]

# Show images
plt.figure()
plt.subplot(1, 3, 1)
show_points(color1, rows1, cols1)
plt.axis("off")
plt.subplot(1, 3, 2)
show_points(color2, rows2, cols2)
plt.axis("off")
plt.subplot(1, 3, 3)
show_points(color3, rows3, cols3)
plt.axis("off")

# Get matched interest points for image pairs
matches_12 = match_interest_points(descriptors1, descriptors2)
matches_13 = match_interest_points(descriptors1, descriptors3)

# Show matched interest points
plt.figure()
plt.subplot(2, 1, 1)
plt.title("Matched interst points for a3p2_0.png and a3p2_1.png")
plt.imshow(np.hstack((color1, color2)))
for idx1, idx2 in matches_12[:15]:
    plt.plot([cols1[idx1], cols2[idx2] + color1.shape[1]], [rows1[idx1], rows2[idx2]], 'xr--')
plt.axis("off")
plt.subplot(2, 1, 2)
plt.title("Matched interst points for a3p2_0.png and a3p2_2.png")
plt.imshow(np.hstack((color1, color3)))
for idx1, idx2 in matches_13[:15]:
    plt.plot([cols1[idx1], cols3[idx2] + color1.shape[1]], [rows1[idx1], rows3[idx2]], 'xr--')
plt.axis("off")
plt.show()


if __name__ == "__main__":
    problem1()
    problem2()