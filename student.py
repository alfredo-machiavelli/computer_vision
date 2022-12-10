
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage import color, transform
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage import filters


def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 

    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 
    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions
    plt.scatter(x, y, zorder=1, color='none', edgecolor='red')
    plt.imshow(image, zorder=0, cmap='gray')  # may need to use extent param
    plt.show()


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image
    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())
    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.
    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions
        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops

    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels
    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image
    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.zeros(1)
    ys = np.zeros(1)
    # Note that xs and ys represent the coordinates of the image. Thus, xs actually denote the columns
    # of the respective points and ys denote the rows of the respective points.

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)

    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    sigma = 2.0
    rescale = False
    if image.shape[0] < 1000 and image.shape[1] < 1000:
        rescale = True
        image = np.float32(transform.rescale(image, 2))

    filtered_image = filters.gaussian(image, sigma=sigma)

    Iy, Ix = np.gradient(filtered_image)
    Ixx = filters.gaussian(Ix * Ix, sigma=sigma)
    Ixy = filters.gaussian(Ix * Iy, sigma=sigma)
    Iyy = filters.gaussian(Iy * Iy, sigma=sigma)

    alpha = 0.04
    R = Ixx * Iyy - Ixy ** 2 - alpha * (Ixx + Iyy) ** 2
    R = R / np.linalg.norm(R)
    threshold = np.percentile(R, [60.0])
    np.putmask(R, R < threshold, 0)
    interested_points = feature.peak_local_max(
        R,
        min_distance=feature_width // 2,
        threshold_abs=2e-6,
        exclude_border=True,
        num_peaks=2000,
    )

    if rescale:
        interested_points = interested_points / 2
    return interested_points[:, 1], interested_points[:, 0]


def get_features(image, x, y, feature_width):
    '''
    Returns features for a given set of interest points.
    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length
    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.
    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.
    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.
    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions
        - skimage.filters (library)
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.
    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the gradient vectors to magnitude and direction.
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors.
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.

    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    if feature_width % 4 != 0:
        raise ValueError("feature_width must be a multiple of 4.")
    x = np.round(x).astype(int).flatten()
    y = np.round(y).astype(int).flatten()
    offset = feature_width // 2
    descriptors = list()
    n_bins = 8

    gradients = np.array(np.gradient(image))
    magnitudes = np.linalg.norm(gradients, axis=0)
    angles = np.angle(np.arctan2(gradients[0], gradients[1]), deg=True)

    for xi, yi in zip(x, y):
        crop_magnitudes = magnitudes[yi-offset +
                                     1:yi+offset+1, xi-offset+1:xi+offset+1]
        crop_angles = angles[yi-offset+1:yi+offset+1, xi-offset+1:xi+offset+1]
        if crop_magnitudes.shape != (feature_width, feature_width):
            continue

        patches_magnitudes = np.array(np.hsplit(np.array(
            np.hsplit(crop_magnitudes, 4)).reshape(4, -1), 4)).reshape(-1, feature_width)
        patches_angles = np.array(np.hsplit(np.array(
            np.hsplit(crop_angles, 4)).reshape(4, -1), 4)).reshape(-1, feature_width)
        feature_vector = list()
        for patch_i in range(patches_magnitudes.shape[0]):
            bins = np.digitize(
                patches_angles[patch_i], np.arange(0, 360, 360 // n_bins))
            bin_vector = np.zeros(n_bins)
            for bin_i in range(0, n_bins):
                mask = np.array(bins == bin_i).flatten()
                bin_vector[bin_i] = np.sum(
                    patches_magnitudes[patch_i].flatten()[mask])
            feature_vector.append(bin_vector)
        feature_vector_norm = (np.array(feature_vector).flatten(
        ) / np.array(feature_vector).flatten().sum())
        descriptors.append(feature_vector_norm)
    return np.array(descriptors)


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.
    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 7.18 in Section 7.1.3 of Szeliski.
    For extra credit you can implement spatial verification of matches.
    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.
    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).
    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2
    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    # These are placeholders - replace with your matches and confidences!

    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: check match_features.pdf
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.

    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    distances = cdist(im1_features, im2_features, metric="euclidean")
    n_rows = distances.shape[0]
    idxs = np.argsort(distances, axis=1)[:, :2]
    d1_ix = [np.arange(n_rows), idxs[:, 0]]
    d2_ix = [np.arange(n_rows), idxs[:, 1]]
    nddr = distances[d1_ix] / distances[d2_ix]

    matches = np.stack((d1_ix[0], d1_ix[1]), axis=1)
    confidences = 1 - nddr

    return matches, confidences
