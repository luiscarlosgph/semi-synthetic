"""
@brief  This file holds classes that store information about the endoscopic images that are
        going to be segmented.
@author Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
@date   25 Aug 2015.
"""

import numpy as np
import os
import cv2
# import caffe
import sys
import random
import matplotlib.pyplot as plt
import scipy.misc
import imutils
import geometry
import tempfile
import PIL
import skimage.morphology
import skimage.util

# My imports
import common


#
# @brief Perlin noise generator.
#
def perlin(x, y, seed):

    # Permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype = int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    # Coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)

    # Internal coordinates
    xf = x - xi
    yf = y - yi

    # Fade factors
    u = fade(xf)
    v = fade(yf)

    # Noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)

    # Combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)

#
# @brief Linear interpolation.
#
def lerp(a, b, x):
    return a + x * (b - a)

#
# @brief 6t^5 - 15t^4 + 10t^3.
#
def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

#
# @brief Grad converts h to the right gradient vector and return the dot product with (x, y).
#
def gradient(h, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:,:, 0] * x + g[:,:, 1] * y

#
# @brief Perlin noise image.
#
# @param[in]  height  Height of the output image.
# @param[in]  width   Width of the output image.
# @param[in]  scale   Higher means smaller blobs.
# @param[in]  minval  The minimum noise value.
# @param[in]  maxval  The maximum noise value.
#
# @returns a 2D numpy array.
def perlin2d_smooth(height, width, scale, minval = 0.0, maxval = 1.0, seed = None):
    lin_y = np.linspace(0, scale, height, endpoint = False)
    lin_x = np.linspace(0, scale, width, endpoint = False)
    x, y = np.meshgrid(lin_x, lin_y)
    arr = perlin(x, y, seed)
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    arr = (np.clip((arr - min_arr) / (max_arr - min_arr), 0.0, 1.0) * (maxval - minval)) + minval
    return arr

#
# @brief Given a set of 2D points it finds the center and radius of a circle.
#
# @param[in] x List or array of x coordinates.
# @param[in] y List or array of y coordinates.
#
# @returns (xc, yc, radius).
def fit_circle(x, y):

    # Coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # Calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # Linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri_1      = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    R_1       = np.mean(Ri_1)
    residu_1  = np.sum((Ri_1-R_1) ** 2)
    residu2_1 = np.sum((Ri_1 ** 2 - R_1 ** 2) ** 2)

    return xc_1, yc_1, R_1

#
# @brief Zero parameter Canny edge detector.
#
def auto_canny(image, sigma = 0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges

#
# @brief Abstract image class. This is not meant to be instantiated and it refers to a general
#        multidimensional image or label.
#
class CaffeinatedAbstract(object):

    #
    # @brief Every image must have at least data and name. We ensure of that with this abstract
    #        constructor that will be called by all the children.
    #
    # @param[in] raw_frame Multidimensional image, at least H x W.
    # @param[in] name      String with the name of the image. It can also be the frame number
    #                      of a video, but it will be converted to string.
    #
    def __init__(self, raw_frame, name):

        # Assert that the frame has data
        if len(raw_frame.shape) <= 1 or raw_frame.shape[0] <= 0 or raw_frame.shape[1] <= 0:
            raise RuntimeError('[CaffeinatedAbstract.__init__], the image provided ' \
                'does not have data.')

        # Assert that the name is valid
        if not name:
            raise ValueError('[CaffeinatedAbstract.__init__] Error, every caffeinated ' \
                'abstract child must have a name.')

        # Store attributes in class
        self._raw_frame = raw_frame
        self._name = str(name)

    #
    # @brief Access to a copy of the internal BGR image.
    #
    # @returns a copy of the internal frame, whatever it is, image or label.
    def raw_copy(self):
        return self._raw_frame.copy()

    #
    # @brief Saves image to file.
    #
    # @param[in] path  Destination path.
    # @param[in] flags Flags that will be passed to OpenCV.
    #
    def save(self, path, flags):

        # Assert that the destination path does not exist
        if common.path_exists(path):
            raise ValueError('[CaffeinatedImage.save] Error, destination path ' \
                + str(path) + ' already exists.')

        if flags:
            return cv2.imwrite(path, self._raw_frame, flags)
        else:
            return cv2.imwrite(path, self._raw_frame)

    #
    # @brief Crops an image in a rectangular fashion, including both corner pixels in the image.
    #
    # @param[in] tlx Integer that represents the top left corner column.
    # @param[in] tly Integer that represents the top left corner row.
    # @param[in] brx Integer that represents the bottom right corner column.
    # @param[in] bry Integer that represents the bottom right corner row.
    #
    # @returns nothing.
    def crop(self, tlx, tly, brx, bry):
        assert(isinstance(tlx, type(0)) and isinstance(tly, type(1)) and isinstance(brx, type(1)) \
            and isinstance(bry, type(1)))
        assert(tlx <= brx)
        assert(tly <= bry)
        self._raw_frame = self._raw_frame[tly:bry + 1, tlx:brx + 1]

    def resize_to_width(self, new_w, interp):
        self._raw_frame = CaffeinatedAbstract.resize_width(self._raw_frame, new_w, interp)

    #
    # @brief Convert binary mask into just the mask of its boundary.
    #
    # @param[in] mask      Input mask.
    # @param[in] thickness Thickness of the border.
    #
    # @returns the boundary mask.
    @staticmethod
    def mask2border(mask, thickness):

        # Find the contour of the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

        # Create a new image with just the contour
        new_mask = np.zeros_like(mask)
        new_mask = cv2.drawContours(new_mask, cnts, -1, 255, thickness)

        return new_mask

    #
    # @brief Histogram equalisation (CLAHE).
    #
    # @param[in] im         Input image.
    # @param[in] clip_limit Contrast limit.
    #
    # @returns the equalised image.
    @staticmethod
    def clahe(im, clip_limit = 2.0):
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
        clahe_engine = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = (8, 8))
        lab[:,:, 0] = clahe_engine.apply(lab[:,:, 0])
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    #
    # @brief Flip left-right.
    #
    # @returns the flipped image.
    @staticmethod
    def fliplr(im):
        return np.fliplr(im)

    #
    # @brief Flip up-down.
    #
    # @returns the flipped image.
    @staticmethod
    def flipud(im):
        return np.flipud(im)

    #
    # @brief Thresholds a grayscale image.
    #
    # @param[in] img    Input grayscale image.
    # @param[in] level  Greater than this level will be set to maxval. Default value is 127.
    # @param[in] maxval Th values greater than level will be set to maxval.
    #                   Default value is 255.
    #
    # @returns the thresholded image.
    @staticmethod
    def bin_thresh(im, level = 127, maxval = 255):
        assert(len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[2] == 1))
        _, thresh = cv2.threshold(np.squeeze(im), level, maxval, cv2.THRESH_BINARY)
        return thresh

    #
    # @brief   Random crop, both dimensions should be equal or smaller than the original size.
    # @details If a list is given, all the images must be larger than the desired new height and
    #          width.
    #
    # @param[in] img        Ndarray with the image, shape (height, width) or
    #                       (height, width, channels).
    # @param[in] new_height Height of the cropped image.
    # @param[in] new_width  Width of the cropped image.
    #
    # @returns a cropped patch.
    @staticmethod
    def random_crop(im, new_height, new_width):
        assert(isinstance(im, np.ndarray))
        assert(new_height > 0 and new_height <= im.shape[0])
        assert(new_width > 0 and new_width <= im.shape[1])

        # Choose random coordinates for crop
        height_border = im.shape[0] - new_height
        width_border = im.shape[1] - new_width
        top_y = random.randint(0, height_border - 1) if height_border > 0 else 0
        top_x = random.randint(0, width_border - 1) if width_border > 0 else 0

        # Crop image
        new_im = im[top_y:top_y + new_height, top_x:top_x + new_width].copy()
        assert(new_im.shape[0] == new_height)
        assert(new_im.shape[1] == new_width)

        return new_im

    #
    # @brief Performs a random crop. New height and width is decided independently, this
    #        function changes the form factor.
    #
    # @param[in] img     Input image, numpy array.
    # @param[in] delta   Minimum factor of change, e.g. if 0.5 the new height and width will be
    #                    minimum half of the original.
    #
    # @returns the new image.
    @staticmethod
    def random_crop_factor(im, delta):
        assert(isinstance(im, np.ndarray))

        min_scale = 1.0 - delta
        max_scale = 1.0
        new_scale = random.uniform(min_scale, max_scale)
        new_height = int(round(im.shape[0] * new_scale))
        new_width = int(round(im.shape[1] * new_scale))
        new_im = CaffeinatedAbstract.random_crop(im, new_height, new_width)

        return new_im

    #
    # @brief Performs a random crop. New height and width is decided independently, this
    #        function changes the form factor.
    #
    # @param[in] img     Input image, numpy array.
    # @param[in] delta   Minimum factor of change, e.g. if 0.5 the new height and width will be
    #                    minimum half of the original.
    #
    # @returns the new image.
    @staticmethod
    def random_crop_no_factor(im, delta):
        assert(isinstance(im, np.ndarray))

        min_scale = 1.0 - delta
        max_scale = 1.0
        new_height = int(round(im.shape[0] * random.uniform(min_scale, max_scale)))
        new_width = int(round(im.shape[1] * random.uniform(min_scale, max_scale)))
        new_im = CaffeinatedAbstract.random_crop(im, new_height, new_width)

        return new_im

    #
    # @brief Random crop of a list of images. The crops will be performed in different locations
    #        for the different images of the list, but all the output images will have the same
    #        size.
    #
    # @param[in] im_list   List of images to be cropped.
    # @param[in] new_height Height of the cropped image.
    # @param[in] new_width  Width of the cropped image.
    #
    # @returns a list of cropped images to the desired size.
    @staticmethod
    def random_crop_list(im_list, new_height, new_width):
        assert(isinstance(im_list, list))
        assert(len(im_list) > 0)

        new_im_list = [ CaffeinatedAbstract.random_crop(im, new_height, new_width) \
            for im in im_list ]

        return new_im_list

    #
    # @brief Random crop all the images of the list in the same coordinates for all of them.
    #        All the input images MUST have the same size.
    #
    # @param[in] im_list   List of images to be cropped.
    # @param[in] new_height Height of the cropped image.
    # @param[in] new_width  Width of the cropped image.
    #
    # @returns a list of cropped images to the desired size.
    @staticmethod
    def random_crop_same_coord_list(im_list, new_height, new_width):
        assert(isinstance(im_list, list))
        assert(len(im_list) > 0)

        # Choose random coordinates for crop
        height_border = im_list[0].shape[0] - new_height
        width_border = im_list[0].shape[1] - new_width
        top_y = random.randint(0, height_border - 1) if height_border > 0 else 0
        top_x = random.randint(0, width_border - 1) if width_border > 0 else 0

        # Crop all the images in the list
        new_im_list = [ im[top_y:top_y + new_height, top_x:top_x + new_width].copy() \
            for im in im_list ]

        return new_im_list

    #
    # @brief Random crop all the images of the list in the same coordinates for all of them.
    #        All the images MUST have the same size. The output images will have the same form
    #        factor.
    #
    # @param[in] im_list List of images to be cropped.
    # @param[in] delta    Minimum factor of change, e.g. if 0.5 the new height and width will be
    #                     minimum half of the original.
    #
    # @returns a list of cropped images to the desired size.
    @staticmethod
    def random_crop_same_coord_list_factor(im_list, delta):
        assert(isinstance(im_list, list))
        assert(len(im_list) > 0)

        # Get the dimensions of the new images
        min_scale = 1.0 - delta
        max_scale = 1.0
        new_scale = random.uniform(min_scale, max_scale)
        new_height = int(round(im.shape[0] * new_scale))
        new_width = int(round(im.shape[1] * new_scale))

        return CaffeinatedAbstract.random_crop_same_coord_list(im_list, new_height, new_width)

    #
    # @brief Random crop all the images of the list in the same coordinates for all of them.
    #        All the images MUST have the same size. The output images will not have the same
    #        form factor.
    #
    # @param[in] im_list List of images to be cropped.
    # @param[in] delta    Minimum factor of change, e.g. if 0.5 the new height and width will be
    #                     minimum half of the original.
    #
    # @returns a list of cropped images to the desired size.
    @staticmethod
    def random_crop_same_coord_list_no_factor(im_list, delta):
        assert(isinstance(im_list, list))
        assert(len(im_list) > 0)

        # Get the dimensions of the new images
        min_scale = 1.0 - delta
        max_scale = 1.0
        new_height = int(round(im_list[0].shape[0] * random.uniform(min_scale, max_scale)))
        new_width = int(round(im_list[0].shape[1] * random.uniform(min_scale, max_scale)))

        return CaffeinatedAbstract.random_crop_same_coord_list(im_list, new_height, new_width)

    #
    # @brief Scale an image keeping original size, that is, the output image will have the
    #        size of the input.
    #
    # @details If the scale factor is smaller than 1.0, the output image will be padded.
    #          Otherwise it will be cropped.
    #
    # @param[in] im           Input image or list of images.
    # @param[in] scale_factor If 1.0, the image stays as it is.
    # @param[in] interp       Method of interpolation: nearest, bilinear, bicubic, lanczos.
    # @param[in] boder_value  Border value. Used when the image is downsized and padded.
    # @param[in] clip_sides   List of sides to crop out. Used only in case the scaling factor
    #                         is lower than 1.0.
    #
    # @returns the scaled image.
    @staticmethod
    def scale_keeping_size(im, scale_factor, interp, border_value, clip_sides = None):
        if clip_sides is None:
            clip_sides = []

        # Resize image to the desired new scale
        new_im = CaffeinatedAbstract.resize_factor(im, scale_factor, interp)

        # If the new image is larger, we crop it
        if new_im.shape[0] > im.shape[0]:
            new_im = CaffeinatedAbstract.crop_center(new_im, im.shape[1], im.shape[0])

        # If the new image is smaller, we pad it
        elif new_im.shape[0] < im.shape[0]:
            padded = np.full_like(im, border_value)
            start_row = (padded.shape[0] // 2) - (new_im.shape[0] // 2)
            start_col = (padded.shape[1] // 2) - (new_im.shape[1] // 2)
            end_row = start_row + new_im.shape[0]
            end_col = start_col + new_im.shape[1]
            padded[start_row:end_row, start_col:end_col] = new_im
            new_im = padded

            # Move the image to the desired sides (used to downscale tools and still keep them
            # attached to the border of the image)
            if 'top' in clip_sides:
                M = np.float32([[1, 0, 0], [0, 1, -start_row]])
                new_im = cv2.warpAffine(new_im, M, (padded.shape[1], padded.shape[0]),
                    interp, cv2.BORDER_CONSTANT, border_value)

            if 'left' in clip_sides:
                M = np.float32([[1, 0, -start_col], [0, 1, 0]])
                new_im = cv2.warpAffine(new_im, M, (padded.shape[1], padded.shape[0]),
                    interp, cv2.BORDER_CONSTANT, border_value)

            if 'bottom' in clip_sides:
                M = np.float32([[1, 0, 0], [0, 1, start_row]])
                new_im = cv2.warpAffine(new_im, M, (padded.shape[1], padded.shape[0]),
                    interp, cv2.BORDER_CONSTANT, border_value)

            if 'right' in clip_sides:
                M = np.float32([[1, 0, start_col], [0, 1, 0]])
                new_im = cv2.warpAffine(new_im, M, (padded.shape[1], padded.shape[0]),
                    interp, cv2.BORDER_CONSTANT, border_value)

        return new_im

    #
    # @brief Could flip the image or not.
    #
    # @param[in] im Image or list of images. If list, all images are either flipped or not.
    #
    # @returns the image (maybe flipped) maybe just the original one.
    @staticmethod
    def random_fliplr(im, not_used = None):
        if common.randbin():
            if isinstance(im, list):
                return [ CaffeinatedAbstract.fliplr(i) for i in im ]
            else:
                return CaffeinatedAbstract.fliplr(im)
        else:
            return im

    #
    # @brief Could flip the image or not.
    #
    # @param[in] im Image or list of images. If list, all images are either flipped or not.
    #
    # @returns the image (maybe flipped) maybe just the original one.
    @staticmethod
    def random_flipud(im, not_used = None):
        if common.randbin():
            if isinstance(im, list):
                return [ CaffeinatedAbstract.flipud(i) for i in im ]
            else:
                return CaffeinatedAbstract.flipud(im)
        else:
            return im

    #
    # @brief Add motion blur in a specific direction.
    #
    # @param[in] im       Input image.
    # @param[in] mask     Pass a foreground mask if you wanna apply the motion just in the
    #                     foreground.
    # @param[in] apply_on Either 'bg', 'fg' or 'both'.
    # @param[in] ks       Size of the convolution kernel to be applied.
    # @param[in] phi_deg  Angle of rotation in degrees. Default is zero, so the motion will be
    #                     horizontal.
    #
    # @returns the blured images.
    @staticmethod
    def directional_motion_blur(im, phi_deg = 0, ks = 15):

        # Generating the kernel
        kernel = np.zeros((ks, ks))
        kernel[int((ks - 1) / 2),:] = np.ones(ks) / ks

        # Rotate image if the user wants to simulate motion in a particular direction
        # rot_im = CaffeinatedAbstract.rotate_bound(im, phi_deg, cv2.INTER_CUBIC)
        # rot_im_blur = cv2.filter2D(rot_im, -1, kernel)
        # new_im = CaffeinatedAbstract.rotate_bound(rot_im_blur, -phi_deg, cv2.INTER_CUBIC)
        # tly = (new_im.shape[0] - im.shape[0]) // 2
        # tlx = (new_im.shape[1] - im.shape[1]) // 2
        # new_im = new_im[tly:tly + im.shape[0], tlx:tlx + im.shape[1]]

        # FIXME: We keep just horizontal motion to investigate drop in performance
        new_im = cv2.filter2D(im, -1, kernel)

        return new_im

    #
    # @brief Random motion blur. Both foreground and background images must have the same size.
    #
    # @param[in] im         Input image.
    # @param[in] mask       Mask of the foreground object that will appear blurred within the
    #                       image.
    # @param[in] rho        Magnitude in pixels of the foreground motion vector.
    # @param[in] phi_deg    Angle in degrees of the motion vector.
    # @param[in] interlaced Random interlacing will be added. Some lines of the foreground will
    #                       move and others will not.
    # @param[in] alpha      Weight for the weighted sum. Default value is 0.5.
    #
    # @returns the blurred image.
    @staticmethod
    def weighted_sum_motion_blur(im, mask, rho, phi_deg, interlaced = False,
            alpha = 0.5):
        assert(im.shape[0] == mask.shape[0])
        assert(im.shape[1] == mask.shape[1])

        # Compute random motion vector
        phi = common.deg_to_rad(phi_deg)
        tx = rho * np.cos(phi)
        ty = rho * np.sin(phi)

        # Translation matrix
        trans_mat = np.eye(3)
        trans_mat[0, 2] = tx
        trans_mat[1, 2] = ty
        mat = trans_mat[:2, :3]

        # Warp current image and mask according to the motion vector
        im_warped = cv2.warpAffine(im, mat, (im.shape[1], im.shape[0]), flags = cv2.INTER_CUBIC)
        mask_warped = cv2.warpAffine(mask, mat, (im.shape[1], im.shape[0]),
            flags = cv2.INTER_NEAREST)

        # Interlacing
        if interlaced:
            mask_warped_orig = mask_warped.copy()
            lines_with_mask = np.unique(np.nonzero(mask_warped)[0]).tolist()
            if lines_with_mask:
                num_lines_to_remove = np.random.randint(len(lines_with_mask))
                random.shuffle(lines_with_mask)
                lines_with_mask = lines_with_mask[:num_lines_to_remove]
                for i in lines_with_mask:
                    mask_warped[i,:] = 0

        # Combine both images
        new_im = im.copy()
        new_im[mask_warped > 0] = np.round(
            alpha * im[mask_warped > 0] + (1. - alpha) * im_warped[mask_warped > 0]
        ).astype(np.uint8)

        # Blur if interlaced
        if interlaced:
            ksize = 3
            blurred = cv2.GaussianBlur(new_im, (ksize, ksize), 0)
            new_im[mask_warped_orig > 0] = blurred[mask_warped_orig > 0]

        return new_im

    #
    # @brief Adds or subtracts intensity in different parts of the image using Perlin noise.
    #
    # @param[in] im Input image.
    #
    # @returns the augmented image.
    @staticmethod
    def random_local_brightness_augmentation(im, intensity_start = 50., intensity_stop = 200.,
            intensity_step = 50., shape_start = 1., shape_stop = 5., shape_step = 1.):

        # Generate random illumination change range
        intensity_options = np.arange(intensity_start, intensity_stop + intensity_step, intensity_step)
        change_choice = np.random.choice(intensity_options)

        # Generate Perlin blob size, larger numbers mean smaller blobs
        shape_options = np.arange(shape_start, shape_stop + shape_step, shape_step)
        shape_choice = np.random.choice(shape_options)

        # Generate Perlin additive noise mask
        pn = perlin2d_smooth(im.shape[0], im.shape[1], shape_choice) * change_choice \
            - .5 * change_choice
        pn = np.dstack((pn, pn, pn))

        # Modify the image: HSV option
        # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float64)
        # hsv[:, :, 2] = np.round(np.clip(hsv[:, :, 2] + pn, 0, 255))
        # augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Additive value on BGR
        augmented = np.round(np.clip(im.astype(np.float64) + pn, 0., 255.)).astype(np.uint8)

        return augmented

    #
    # @brief Adds or subtracts intensity in different parts of the image using Perlin noise.
    #
    # @param[in] im Input image.
    #
    # @returns the augmented image.
    @staticmethod
    def random_local_contrast_augmentation(im, shape_start = 1., shape_stop = 5., shape_step = 1.):

        # Choose minimum and maximum contrast randomly
        contrast_min = random.choice([0.5, 0.6, 0.7, 0.8])
        contrast_max = random.choice([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

        # Generate Perlin blob size, larger numbers mean smaller blobs
        shape_options = np.arange(shape_start, shape_stop + shape_step, shape_step)
        shape_choice = np.random.choice(shape_options)

        # Generate Perlin additive noise mask
        pn = perlin2d_smooth(im.shape[0], im.shape[1], shape_choice, minval = contrast_min,
            maxval = contrast_max)
        pn = np.dstack((pn, pn, pn))

        # Modify the image
        augmented = np.round(np.clip(np.multiply(im.astype(np.float64), pn), 0, 255)).astype(np.uint8)

        return augmented

    #
    # @brief Global (as in same additive value added to all pixels) brightness augmentation.
    #
    # @param[in] im Input image.
    #
    # @returns the augmented image.
    @staticmethod
    def random_global_brightness_augmentation(im, intensity_start = -50, intensity_stop = 50,
            intensity_step = 10):

        # Generate random illumination change
        intensity_options = np.arange(intensity_start, intensity_stop + intensity_step,
            intensity_step)
        change_choice = np.random.choice(intensity_options)

        # Additive change on Value of HSV
        # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float64)
        # hsv[:, :, 2] = np.round(np.clip(hsv[:, :, 2] + change_choice, 0., 255.))
        # hsv = hsv.astype(np.uint8)
        # augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Additive change on all channels of BGR
        augmented = np.clip(im.astype(np.float64) + change_choice, 0, 255).astype(np.uint8)

        return augmented

    #
    # @brief Global contrast (multiplicative) augmentation.
    #
    # @param[in] im Input image.
    #
    # @returns the augmented image.
    @staticmethod
    def random_global_contrast_augmentation(im):
        contrast_choice = random.choice([0.5, 0.6, 0.7, 0.8, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
            1.9, 2.0])
        augmented = np.round(np.clip(np.multiply(im.astype(np.float64),
            contrast_choice), 0, 255)).astype(np.uint8)

        return augmented

    #
    # @brief Bernoulli motion blur.
    #
    # @param[in] im      Image or list of images.
    # @param[in] mask    Mask of moving object.
    # @param[in] max_mag Maximum amount of pixels of displacement.
    # @param[in] max_ang Maximum angle of the motion vector. Default is 360, i.e. can move in any
    #                    direction.
    #
    # @returns the images with motion blur with a probability p.
    @staticmethod
    def random_weighted_sum_motion_blur(im, mask, max_mag = 32, max_ang = 360):
        rho = np.random.randint(max_mag)
        phi_deg = np.random.randint(max_ang)
        interlaced = common.randbin()
        if isinstance(im, list):
            return [ CaffeinatedAbstract.weighted_sum_motion_blur(i, m, rho, phi_deg,
                interlaced) for i, m in zip(im, mask) ]
        else:
            return CaffeinatedAbstract.weighted_sum_motion_blur(im, mask, rho, phi_deg,
                interlaced)

    #
    # @brief Converts an image from BGR to BRG.
    #
    # @param[in] im BGR image.
    #
    # @returns an image converted to BRG.
    @staticmethod
    def bgr2brg(im):
        return im[..., [0, 2, 1]]

    #
    # @brief Bernoulli BGR to BRG swapping.
    #
    # @param[in] im Image or list of images.
    #
    # @returns the image with the green-red channels swapped with a probability of 0.5.
    @staticmethod
    def random_brg(im):
        if common.randbin():
            if isinstance(im, list):
                return [ CaffeinatedAbstract.bgr2brg(i) for i in im ]
            else:
                return CaffeinatedAbstract.bgr2brg(im)
        else:
            return im

    #
    # @brief Rotates the image over itself a random number of degrees.
    #
    # @param[in] im       Input image, numpy array.
    # @param[in] deg_delta The range of possible rotation is +- deg_delta.
    # @param[in] interp    Interpolation method: lanczos, linear, cubic, nearest.
    #
    # @returns the rotated image.
    @staticmethod
    def random_rotation(im, deg_delta, interp):
        max_ang = deg_delta
        min_ang = -1. * max_ang
        ang = random.uniform(min_ang, max_ang)
        new_im = None
        if isinstance(im, list):
            new_im = [ CaffeinatedAbstract.rotate_and_crop(i, ang, interp) for i in im ]
        else:
            new_im = CaffeinatedAbstract.rotate_and_crop(im, ang, interp)
        return new_im

    #
    # @brief Resizes an imaged to the desired width while keeping proportions.
    #
    # @param[in] im     Image to be resized.
    # @param[in] new_w  New width.
    # @param[in] interp Method of interpolation: nearest, bilinear, bicubic, lanczos.
    #
    # @returns a resized image.
    @staticmethod
    def resize_width(im, new_w, interp = None):
        assert(im.dtype == np.uint8)

        # If no interpolation method is chosen we select the most convenient depending on whether
        # the user is upsampling or downsampling the image
        if interp is None:
            interp = cv2.INTER_AREA if new_w < im.shape[1] else cv2.INTER_LANCZOS4

        ratio = float(im.shape[0]) / float(im.shape[1])
        new_h = int(round(new_w * ratio))
        new_im = cv2.resize(im, (new_w, new_h), interpolation=interp)

        return new_im

    #
    # @brief Resizes an imaged to the desired width while keeping proportions.
    #
    # @param[in] im     Image to be resized.
    # @param[in] new_h  New height.
    # @param[in] interp Method of interpolation: nearest, bilinear, bicubic, lanczos.
    #
    # @returns a resized image.
    @staticmethod
    def resize_height(im, new_h, interp):
        assert(im.dtype == np.uint8)
        ratio = float(im.shape[0]) / float(im.shape[1])
        new_w = int(round(new_h / ratio))
        # imethod = PIL_interp_method[interp]
        # new_im = np.array(PIL.Image.fromarray(im).resize((new_w, new_h), imethod))
        new_im = cv2.resize(im, (new_w, new_h), interpolation=interp)
        return new_im

    #
    # @brief Scales an image to a desired factor of the original one.
    #
    # @param[in] im           Image to be resized.
    # @param[in] scale_factor Factor to scale up or down the image.
    # @param[in] interp       Method of interpolation: nearest, bilinear, bicubic, lanczos.
    #
    # @returns a resized image.
    @staticmethod
    def resize_factor(im, scale_factor, interp):
        new_w = int(round(im.shape[1] * scale_factor))
        return CaffeinatedAbstract.resize_width(im, new_w, interp)

    #
    # @brief Scales an image to a desired factor of the original one.
    #
    # @param[in] im     Image to be resized.
    # @param[in] new_w  New width.
    # @param[in] new_h  New width.
    # @param[in] interp Method of interpolation: nearest, bilinear, bicubic, lanczos.
    #
    # @returns a resized image.
    @staticmethod
    def resize(im, new_w, new_h, interp):
        # imethod = PIL_interp_method[interp]
        # new_im = scipy.misc.imresize(im, (new_h, new_w), interp = interp).astype(im.dtype)
        # return np.array(PIL.Image.fromarray(im).resize((new_w, new_h), imethod),
        #    dtype = im.dtype)
        new_im = cv2.resize(im, (new_w, new_h), interpolation=interp)
        return new_im

    #
    # @returns a crop of shape (new_h, new_w).
    #
    @staticmethod
    def crop_center(im, new_w, new_h):
        start_x = im.shape[1] // 2 - (new_w // 2)
        start_y = im.shape[0] // 2 - (new_h // 2)
        return im[start_y:start_y + new_h, start_x:start_x + new_w].copy()

    #
    # @brief Rotatation of an image with black bounds around it, as it would be
    #        expected. A positive rotation angle results in a clockwise rotation.
    #
    # @param[in] image  Numpy ndarray.
    # @param[in] angle  Angle in degrees.
    #
    # @returns the rotated image.
    @staticmethod
    def rotate_bound(image, angle, interp):

        # Grab the dimensions of the image and then determine the center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # Grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # Perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), flags = interp)

    #
    # brief Rotates an image over a centre point given and leaves the whole
    #       image inside. Clockwise rotation of the image.
    #
    # @param[in] im  Numpy ndarray.
    # @param[in] centre (x, y) in image coordinates.
    # @param[in] angle  Angle in degrees.
    # @param[in] interp OpenCV interpolation method.
    @staticmethod
    def rotate_bound_centre(im, centre, deg, interp):
        cm_x = centre[0]
        cm_y = centre[1]

        # Build the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((cm_y, cm_x), -deg, 1.0)
        rot_mat_hom = np.zeros((3, 3))
        rot_mat_hom[:2,:] = rot_mat
        rot_mat_hom[2, 2] = 1

        # Find the coordinates of the corners in the rotated image
        h = im.shape[0]
        w = im.shape[1]
        tl = np.array([0, 0, 1]).reshape((3, 1))
        tr = np.array([w - 1, 0, 1]).reshape((3, 1))
        bl = np.array([0, h - 1, 1]).reshape((3, 1))
        br = np.array([w - 1, h - 1, 1]).reshape((3, 1))
        tl_rot = np.round(np.dot(rot_mat_hom, tl)).astype(np.int)
        tr_rot = np.round(np.dot(rot_mat_hom, tr)).astype(np.int)
        bl_rot = np.round(np.dot(rot_mat_hom, bl)).astype(np.int)
        br_rot = np.round(np.dot(rot_mat_hom, br)).astype(np.int)

        # Compute the size of the new image from the coordinates of the rotated one so that
        # we add black bounds around the rotated one
        min_x = min([tl_rot[0], tr_rot[0], bl_rot[0], br_rot[0]])
        max_x = max([tl_rot[0], tr_rot[0], bl_rot[0], br_rot[0]])
        min_y = min([tl_rot[1], tr_rot[1], bl_rot[1], br_rot[1]])
        max_y = max([tl_rot[1], tr_rot[1], bl_rot[1], br_rot[1]])
        new_w = max_x + 1 - min_x
        new_h = max_y + 1 - min_y

        # Correct the translation so that the rotated image lies inside the window
        rot_mat[0, 2] -= min_x
        rot_mat[1, 2] -= min_y

        return cv2.warpAffine(im, rot_mat, (new_w[0], new_h[0]), flags = interp)

    #
    # @brief Clockwise rotation plus crop (so that there is no extra added black background).
    #
    # @details The crop is done based on a rectangle of maximal area inside the rotated region.
    #
    # @param[in] im     Numpy ndarray image. Shape (h, w, 3) or (h, w).
    # @param[in] ang    Angle in degrees.
    # @param[in] interp Interpolation method: lanczos, linear, cubic, nearest.
    #
    # @returns the rotated image.
    @staticmethod
    def rotate_and_crop(im, ang, interp):

        # Rotate image
        rotated = CaffeinatedAbstract.rotate_bound(im, ang, interp)

        # Calculate cropping area
        wr, hr = geometry.rotated_rect_with_max_area(im.shape[1],
            im.shape[0], common.deg_to_rad(ang))
        wr = int(np.floor(wr))
        hr = int(np.floor(hr))

        # Centre crop
        rotated = CaffeinatedAbstract.crop_center(rotated, wr, hr)

        return rotated

    #
    # @brief This method deinterlaces an image using ffmpeg.
    #
    # @param[in] im  Numpy ndarray image. Shape (h, w, 3) or (h, w).
    #
    # @returns the deinterlaced image.
    @staticmethod
    def deinterlace(im, ext = '.png'):

        input_path = tempfile.gettempdir() + '/' + common.gen_rand_str() + ext
        output_path = tempfile.gettempdir() + '/' + common.gen_rand_str() + ext

        # Save image in a temporary folder
        cv2.imwrite(input_path, im)

        # Deinterlace using ffmpeg
        common.shell('ffmpeg -i ' + input_path + ' -vf yadif ' + output_path)

        # Read deinterlaced image
        dei = cv2.imread(output_path)

        # Remove image from temporary folder
        common.rm(input_path)
        common.rm(output_path)

        return dei

    @staticmethod
    def gaussian_noise(im, mean=0, std=20):
        noise = np.random.normal(mean, std, im.shape)
        return np.round(np.clip(im.astype(np.float64) + noise, 0, 255)).astype(np.uint8)

    #
    # @rteurns a gamma corrected image.
    #
    @staticmethod
    def adjust_gamma(im, gamma = 1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(im, table)

    #
    # @brief Draws an horizontal gradient image.
    #
    # @returns the image of the gradient.
    @staticmethod
    def draw_grad_lr(height, width, left_colour, right_colour):
        return (np.ones((height, width)) * np.linspace(left_colour, right_colour,
            width)).astype(np.uint8)

    #
    # @brief Draws an horizontal gradient image.
    #
    # @returns the image of the gradient.
    @staticmethod
    def draw_grad_ud(height, width, left_colour, right_colour):
        return (np.ones((height, width)) * np.linspace(left_colour, right_colour,
            width)).astype(np.uint8).T

    #
    # @brief FIXME: does not work properly when image is dark
    @staticmethod
    def detect_endoscopic_circle_bbox(im):

        # Edge detection
        max_black_intensity = 10
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(gray, kernel, iterations = 1)
        _, thresh = cv2.threshold(dilation, max_black_intensity, 255, cv2.THRESH_BINARY)

        # Detect contour of largest area
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = cv2.contourArea)
        ((xc, yc), radius) = cv2.minEnclosingCircle(cnt)
        x = xc - radius
        y = yc - radius
        w = 2 * radius
        h = 2 * radius
        # x, y, w, h = cv2.boundingRect(cnt)

        return int(x), int(y), int(w), int(h)

    @staticmethod
    def crop_endoscopic_circle(im):

        # Detect endoscopic circle
        has_circle = True
        # TODO
        if not has_circle:
            return im
        x, y, w, h = CaffeinatedAbstract.detect_endoscopic_circle_bbox(im)
        cropped = im[y:y + h, x:x + w].copy()

        return cropped

    #
    # @brief Function to add specular reflections to an image.
    #
    # TODO
    #
    #
    @staticmethod
    def add_specular_noise():
        pass

    #
    # @brief Skeletonisation of a binary image [0, 255].
    #
    # @param[in]  im        Input binary image. Binary means either some values are zero and some
    #                       others different from zero. Different from 0 can be 1 and 255.
    #
    # @returns a binary image (0, 255) with the skeleton of the image.
    @staticmethod
    def skeleton(im):
        assert(len(im.shape) == 2)
        sk = skimage.morphology.skeletonize_3d(im.astype(bool))
        return sk

    #
    # @brief Pads and image with extra pixels according to a newly specified size.
    #
    # @param[in] tlx       Integer that represents the top left corner column.
    # @param[in] tly       Integer that represents the top left corner row.
    # @param[in] brx       Integer that represents the bottom right corner column.
    # @param[in] bry       Integer that represents the bottom right corner row.
    # @param[in] width     Width of the new image.
    # @param[in] height    Height of the new image.
    # @param[in] intensity Integer of the padding pixels.
    #
    # @returns nothing.
    def pad(self, tlx, tly, brx, bry, width, height, intensity):
        assert(isinstance(tlx, type(0)) and isinstance(tly, type(1)) and isinstance(brx, type(1)) \
            and isinstance(bry, type(1)))
        assert(tlx <= brx)
        assert(tly <= bry)
        assert(width >= self.width)
        assert(height >= self.height)
        assert(isinstance(intensity, type(1)))

        # Create image of the new size
        new_raw_frame = None
        new_pixel = None
        if len(self._raw_frame.shape) == 2:
            new_raw_frame = np.empty((height, width), dtype=self._raw_frame.dtype)
            new_pixel = intensity
        elif len(self._raw_frame.shape) == 3:
            new_raw_frame = np.empty((height, width, self._raw_frame.shape[2]),
                dtype=self._raw_frame.dtype)
            new_pixel = np.empty((self._raw_frame.shape[2],), dtype=self._raw_frame.dtype)
            new_pixel.fill(intensity)
        else:
            raise ValueError('[image.CaffeinatedAbstract.pad] Error, image dimension ' \
                + str(self._raw_frame.shape) + ' not supported.')
        new_raw_frame[:,:] = new_pixel

        # Insert the previous image in the right place
        new_raw_frame[tly:bry + 1, tlx:brx + 1] = self._raw_frame
        self._raw_frame = new_raw_frame

    #
    # @brief Converts the image into a distance transform (L2 norm) to the edges.
    #
    # @param[in] mask_size Size of the Sobel filter kernel.
    #
    # @returns nothing.
    def shape_transform(self, mask_size):
        assert(isinstance(mask_size, type(0)))

        # Convert to grayscale
        # gray = cv2.cvtColor(self._raw_frame, cv2.COLOR_BGR2GRAY)

        # Sobel filter
        sobel_x_64f = np.absolute(cv2.Sobel(self._raw_frame, cv2.CV_64F, 1, 0, ksize=mask_size))
        sobel_y_64f = np.absolute(cv2.Sobel(self._raw_frame, cv2.CV_64F, 0, 1, ksize=mask_size))
        sobel_64f   = (sobel_x_64f + sobel_y_64f)
        scaled_sobel = np.uint8(255 * sobel_64f / np.max(sobel_64f))

        # Dilate borders
        kernel = np.ones((mask_size, mask_size), np.uint8)
        dilated = cv2.dilate(scaled_sobel, kernel, iterations=1)

        # Threshold
        _, thresh = cv2.threshold(dilated, 1, 255, cv2.THRESH_BINARY)

        # Distance transform
        dist = 255 - (cv2.distanceTransform(255 - thresh, cv2.DIST_L2, maskSize=0))

        # Remove backbround
        dist[self._raw_frame == 0] = 0

        self._raw_frame = dist

    #
    # @brief Converts image to single channel.
    #
    # @returns nothing.
    def convert_to_single_chan(self):
        assert(len(self._raw_frame.shape) == 3)

        # Sanity check: assert that all the pixels of the image have the same intensity value in all the
        # channels
        for channel in range(1, self._raw_frame.shape[2]):
            if not np.array_equal(self._raw_frame[:,:, channel], self._raw_frame[:,:, 0]):
                raise RuntimeError('[CaffeinatedAbstract] Error, the image ' + self._name + ' has ' \
                    + 'channels that are different from each other so it is not clear ' \
                    + 'how to convert it to a proper single channel image.')

        self._raw_frame = self._raw_frame[:,:, 0]

    #
    # @brief Changes the intensity of all the pixels in all the channels to zero.
    #
    # @returns nothing.
    def convert_to_black(self):
        self._raw_frame.fill(0)

    #
    # @brief Filter image with ground truth label, background pixels on the ground truth will be blacked.
    #
    # @param[in] caffe_label CaffeinatedLabel.
    #
    def filter_with_gt(self, caffe_label):
        self._raw_frame[caffe_label.raw == 0] = 0

    #
    # @brief   Builds an object of type CaffeinatedAbstract from an image file.
    #
    # @param[in] path Path to the image file.
    #
    @classmethod
    def from_file(cls, path, *args):
        # return cls(cv2.imread(path, cv2.IMREAD_COLOR), *args)
        return cls(cv2.imread(path, cv2.IMREAD_UNCHANGED), *args)

    #
    # @returns the height of the image.
    #
    @property
    def height(self):
        return self._raw_frame.shape[0]

    #
    # @returns the width of the image.
    #
    @property
    def width(self):
        return self._raw_frame.shape[1]

    #
    # @returns the name of the image.
    #
    @property
    def name(self):
        return self._name

    #
    # @returns the raw internal image.
    #
    @property
    def raw(self):
        return self._raw_frame

    #
    # @returns the data type.
    #
    @property
    def dtype(self):
        return self._raw_frame.dtype

#
# @class CaffeinatedImage represents an image that will be used by Caffe so this class should
#                         provide methods to adapt the original image to the type of input
#                         Caffe is expecting.
#
class CaffeinatedImage(CaffeinatedAbstract):

    #
    # @brief Saves the colour image as an attribute of the class.
    #
    # @param[in] raw_frame Numpy array with a image, shape (h, w) or (h, w, c).
    # @param[in] name      Id of the image, either the name or the frame number, it will be converted to
    #                      str.
    # @param[in] label     Id of the class to whom the image belongs. Only used in case the image is used
    #                      for classification purposes. Default value is None.
    #
    def __init__(self, raw_frame, name, label = None):

        # Assert that the image is multi-channel
        dim = len(raw_frame.shape)
        if dim < 2:
            raise RuntimeError('[CaffeinatedImage.__init__], the image provided has [' + \
                str(dim) + '] dimensions, only (H x W x C) and (H x W) are supported.')

        # Assert that the type of label is correct (i.e. integer) when it is not None
        if label is not None:
            assert(isinstance(label, type(0)))
        self._label = label

        # Call CaffeinatedAbstract constructor
        super(CaffeinatedImage, self).__init__(raw_frame if dim > 2 else np.expand_dims(raw_frame, axis = 2),
            name)

    #
    # @brief   Builds an object of type CaffeinatedImage from file.
    #
    # @details Only supports 3-channel colour images. It will raise errors for images with a different
    #          number of channels.
    #
    # @param[in] path Path to the image file.
    # @classmethod
    # def from_file(cls, path, name):
    #   return cls(cv2.imread(path, cv2.IMREAD_UNCHANGED), name)

    #
    # @brief   Convert image to caffe test input, transposing it to the Caffe format (C x H x W) and
    #          subtracting the training mean.
    #
    # @details The mean needs to be subtracted because there is no transform_param section in the input
    #          layer of the test network.
    #
    # @param[in] mean_values Numpy ndarray with the per channel mean of the training set.
    #                        Shape (channels,).
    #
    # @returns an image ready to be processed by Caffe.
    def convert_to_caffe_input(self, mean_values):
        # Sanity check: the mean values should be equal to the number of channels of the input image
        dim = len(self._raw_frame.shape)
        no_mean_values = mean_values.shape[0]
        if dim < 3: # 1D or 2D images should have only one channel mean
            if no_mean_values != 1:
                raise ValueError('[convert_to_caffe_input] Error, [' + str(no_mean_values) + '] mean ' + \
                    ' values provided, but the image is only 1D or 2D, so only one mean value is required.')
        elif dim == 3:
            channels = self._raw_frame.shape[-1]
            if channels != no_mean_values:
                raise ValueError('[convert_to_caffe_input] Error, [' + str(no_mean_values) \
                    + '] mean values have been provided but the given image has [' + str(channels) \
                    + '] channels.')
        else:
            raise ValueError('[convert_to_caffe_input] Error, high dimensional image not supported.')

        return np.transpose(self._raw_frame.astype(np.float32) - mean_values, (2, 0, 1))

    #
    # @brief Resize the image to the desired new width and height.
    #
    # @param[in] new_h New height.
    # @param[in] new_w New width.
    #
    # @returns nothing.
    def resize(self, new_h, new_w):
        self._raw_frame = cv2.resize(self._raw_frame, (new_w, new_h))

    #
    # @brief Resize the image and keep the original aspect ratio, padding if required.
    #
    # @param[in] new_h Height of the new image.
    # @param[in] new_w Width of the new image.
    #
    # @returns nothing.
    def resize_keeping_aspect(self, new_h, new_w):

        # Store aspect ratio, width and height about the previous dimensions
        w = self.width
        h = self.height
        ar = float(w) / float(h)

        # Create new frame respecting the desired new dimensions
        new_frame = np.zeros((new_h, new_w, self._raw_frame.shape[2]), self._raw_frame.dtype)

        # We scale the larger size of the image and adapt the other one to the aspect ratio
        temp_w = None
        temp_h = None
        y_start = 0
        x_start = 0
        if w >= h:
            temp_w = new_w
            temp_h = int(temp_w / ar)
            y_start = int((new_h - temp_h) / 2.0)
        else:
            temp_h = new_h
            temp_w = int(temp_h * ar)
            x_start = int((new_w - temp_w) / 2.0)

        # We add black padding if there is free space
        new_frame[y_start:temp_h + y_start, x_start:temp_w + x_start] = cv2.resize(self._raw_frame,
            (temp_w, temp_h))

        # Copy the final image to the internal buffer that will be displayed
        self._raw_frame = new_frame

    #
    # @brief   Converts BGR image to a Caffe datum with shape (C x H x W).
    #
    # @returns the Caffe datum serialised as a string.
    def serialise_to_string(self, jpeg_quality=100):
        assert(self._raw_frame.dtype == np.uint8)

        import caffe
        # caffe_image = self._raw_frame.astype(np.float32)

        # Convert image to Caffe datum
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height, datum.width, datum.channels = self._raw_frame.shape
        # datum.data = caffe_image.tostring()
        flags = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        datum.data = cv2.imencode('.jpg', self._raw_frame, flags)[1].tostring()

        # If the image has a label, it must be an integer
        if self._label is not None:
            assert(isinstance(self._label, type(0)))
            datum.label = self._label

        return datum.SerializeToString()

    #
    # @brief Convert image from uint16 to uint8.
    #
    def uint16_to_uint8(self):
        self._raw_frame = np.round((self._raw_frame.astype(np.float32) / 65535.0) * 255.0).astype(np.uint8)

    #
    # @brief Add Gaussian noise to image.
    #
    # @param[in] mean Default value is 0.
    # @param[in] std  Default value is 10.
    #
    # @returns nothing.
    def add_gaussian_noise(self, mean = 0, std = 10):

        # Get image dimensions
        row, col, ch = self._raw_frame.shape

        # Add Gaussian noise to the internal image
        gauss = np.random.normal(mean, std, (row, col, ch)).reshape(row, col, ch)

        # Convert image to float, add Gaussian noise and convert back to uint8
        self._raw_frame = np.round(self._raw_frame.astype(np.float64) + gauss).astype(np.uint8)

    #
    # @brief Converts a green screen image with tools to grayscale
    #        adding a bit of noise so that BGR are not kept equal.
    #
    @classmethod
    def gray_tools(cls, im, noise_delta=3):
        assert(isinstance(im, np.ndarray))
        new_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        new_im = cv2.cvtColor(new_im, cv2.COLOR_GRAY2BGR)
        noise = np.random.randint(-noise_delta,
                noise_delta + 1, size=new_im.shape)
        new_im = np.clip(new_im + noise, 0, 255).astype(np.uint8)
        return new_im

    #
    # @brief Convert it to a noisy grayscale image.
    #
    def noisy_gray(self, noise_delta=3):
        self._raw_frame = CaffeinatedImage.gray_tools(self._raw_frame, noise_delta)

    def random_crop(self, height, width):
        self._raw_frame = CaffeinatedAbstract.random_crop(self._raw_frame, height, width)

    @property
    def shape(self):
        return self._raw_frame.shape

#
# @class Caffeinated8UC3Image represents a colour (H x W x 3) CaffeinatedImage.
#
class Caffeinated8UC3Image(CaffeinatedImage):
    #
    # @brief Saves the colour image as an attribute of the class.
    #
    # @param[in] frame_bgr Numpy array with a BGR image, shape (h, w, c).
    def __init__(self, frame_bgr, name):

        # Check that it is a 3-channel BGR image
        EXPECTED_DIM = 3
        EXPECTED_CHANNELS = 3
        if len(frame_bgr.shape) != EXPECTED_DIM or frame_bgr.shape[EXPECTED_DIM - 1] != EXPECTED_CHANNELS:
            raise RuntimeError('[Caffeinated8UC3Image] Error, the image provided has a shape of ' + \
                str(frame_bgr.shape) + '. We expect an image of shape (H x W x ' + \
                str(EXPECTED_CHANNELS) + ').')

        # Check that the image is uint8
        EXPECTED_TYPE = np.uint8
        if frame_bgr.dtype != EXPECTED_TYPE:
            raise RuntimeError('[Caffeinated8UC3Image] Error, the image provided has a type of ' + \
            str(frame_bgr.dtype) + ' and we expect ' + str(EXPECTED_TYPE) + '.')

        super(self.__class__, self).__init__(frame_bgr, name)

#
# @class CaffeinatedLabel represents a segmentation label that will be used by Caffe so this
#                         class should provide methods to adapt the original image to the type of
#                         input Caffe is expecting.
#
# @details This class does not support labels that are not grayscale or colour images, that is,
#          the images provided must be (H x W) or (H x W x C). In case that you provide a label
#          with shape (H x W x C) this class will make sure that all the channels C have the same
#          values. This is because a priori it does not make any sense for a pixel to belong to
#          different classes.
class CaffeinatedLabel(CaffeinatedAbstract):

    #
    # @brief   Stores the label and checks that both dimensions and type are correct for a label.
    # @details To make a safe conversion to single channel this method will check that all the
    #          pixels of the image have exactly the same intensity value in all the BGR channels.
    #          If this does not happen an exception will be raised.
    #
    # @param[in] label_image Single channel OpenCV/Numpy image. Shape (H x W) or (H x W x C).
    # @param[in] name        Name of the label, usually stores the id of the related image.
    # @param[in] classes     Integer that represents the maximum number of classes in the labels,
    #                        used for both validation purposes and to convert back/forth to Caffe
    #                        input.
    # @param[in] class_map   Integer (pixel intensity) -> Integer (class, [0, K - 1]),
    #                        where K is the maximum number of classes.
    # @param[in] proba_map   Probability maps for all the classes, shape (c, h, w).
    #
    def __init__(self, label_image, name, classes, class_map, proba_map = None):

        # This is 2 because we expect the image to be of shape (H x W) and the intensity of the
        # pixel to indicate the class that the pixel belongs to
        EXPECTED_DIM = 2
        EXPECTED_LABEL_TYPE = np.uint8

        # Store the maximum number of classes after validating that it is in the range [2, 256]
        assert(isinstance(classes, type(0)) and classes >= 2 and classes <= 256)
        self._classes = classes

        # Store the dictionary for class mappings after validating it
        classes_present = [False] * classes
        assert(len(class_map.keys()) == classes)
        for k, v in class_map.items():
            assert(isinstance(k, type(0)))
            assert(isinstance(v, type(0)))
            assert(k >= 0 and k <= 255)
            assert(v >= 0 and v < self._classes)
            classes_present[v] = True
        assert(all(classes_present))
        self._class_map = class_map

        # Sanity check: labels that are neither (H x W) nor (H x W x C) are not supported
        dim = len(label_image.shape)
        if not (dim == 2 or dim == 3):
            raise RuntimeError('[CaffeinatedLabel] Error, the label provided has a dimension of ' + \
                str(dim) + ', which is not supported. Only (H x W) and (H x W x C) are supported.')

        # Sanity check: if the label provided is multiple-channel, assert that all the pixels of the image
        #               have the same intensity value in all the channels
        if dim > EXPECTED_DIM:
            for channel in range(1, label_image.shape[2]):
                if not np.array_equal(label_image[:,:, channel], label_image[:,:, 0]):
                    raise RuntimeError('[CaffeinatedLabel] Error, the label provided in ' + name + ' has channels that are ' + \
                        'different from each other so it is not clear how to convert it to a proper '         + \
                        'single channel label in which the intensity defines the pixel class.')

        # Sanity check: the image must be uint8, this essentially means that there is a maximum of 256 labels
        if label_image.dtype != EXPECTED_LABEL_TYPE:
            raise RuntimeError('[CaffeinatedLabel] Error, a label must be ' + str(EXPECTED_LABEL_TYPE) + '.')

        # If the image has several channels, we just get one (we already know that all the channels have the
        # same values
        if dim == EXPECTED_DIM:
            raw_label = label_image
        else:
            raw_label = label_image[:,:, 0]

        # Assert that there are no more unique labels than classes
        unique_classes = np.unique(raw_label)
        if unique_classes.shape[0] > self._classes:
            raise ValueError('[CaffeinatedLabel] Error, label ' + str(name) + ' is said to have ' \
                + str(self._classes) + ' classes but there are more unique values in it, exactly: ' \
                + str(unique_classes))

        # Assert thate the intensities in the label are all present in the class_map dictionary
        for i in unique_classes:
            if not i in self._class_map:
                raise ValueError('[CaffeinatedLabel] Error, label ' + str(name) + ' has a pixel with ' \
                    + 'intensity ' + str(i) + ' but this intensity is not present in the class map.')

        # Store probability map if provided
        if proba_map is not None:
            assert(len(proba_map.shape) == 3)
            assert(proba_map.shape[0] == classes)
            assert(proba_map.shape[1] == raw_label.shape[0])
            assert(proba_map.shape[2] == raw_label.shape[1])
        self._predicted_map = proba_map

        # Call CaffeinatedAbstract constructor
        super(CaffeinatedLabel, self).__init__(raw_label, name)

    #
    # @brief Builds an object of type CaffeinatedLabel from an image file.
    #
    # @param[in] fmaps       array_like, shape (c, h, w).
    #
    # @param[in] classes     Integer that represents the maximum number of classes in the labels, used for
    #                        both validation purposes and to convert back/forth to Caffe input.
    #
    # @param[in] class_map   Integer (pixel intensity) -> Integer (class, [0, K - 1]), where K is the
    #                        maximum number of classes.
    #
    @classmethod
    def from_network_output(cls, fmaps, name, classes, class_map):
        label_image = fmaps.argmax(axis=0).astype(np.uint8)
        for k, v in class_map.items():
            label_image[label_image == v] = k
        return cls(label_image, name, classes, class_map, fmaps)

    #
    # @brief Convert label to CaffeinatedImage for displaying purposes.
    #
    # @param[in] cn Channels of the new image. The labels will be replicated across channels.
    #
    # @returns the label converted into a cn-channel CaffeinatedImage.
    def to_image(self, cn = 3):
        new_image = np.ndarray((self._raw_frame.shape[0], self._raw_frame.shape[1], cn),
            self._raw_frame.dtype)

        for k in range(cn):
            new_image[:,:, k] = self._raw_frame

        return CaffeinatedImage(new_image, self._name)

    #
    # @brief Converts the label to a Caffe datum.
    #
    # @returns a Caffe datum label serialised to string.
    def serialise_to_string(self):

        # Sanity check: assert that the type of the label is correct
        import caffe
        assert(self._raw_frame.dtype == np.uint8)

        # Create Caffe datum
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height, datum.width = self._raw_frame.shape

        # if self._classes == 2:
            # Convert (h, w) -> (1, h, w)
            # caffe_label = np.expand_dims(self._raw_frame, axis = 0)
            # caffe_label = self._raw_frame
        # else:

        # Create ndarray of binary maps
        fmaps = np.zeros([self._classes, self._raw_frame.shape[0], self._raw_frame.shape[1]],
            dtype = np.uint8)

        # k is intensity
        # v is the class number
        for k, v in self._class_map.items():
            fmaps[v, self._raw_frame == k] = 1

        # if self._classes == 2:
        # Binary case, only one feature map
        #    datum.channels = 1
        #    caffe_label = np.expand_dims(fmaps[1], axis = 0)
        # else:
        # Multi-class case, one feature map per class
        #    datum.channels = self._classes
        #    caffe_label = fmaps

        # Multi-class case, one feature map per class
        datum.channels = self._classes
        caffe_label = fmaps

        # Convert label[s] to string
        datum.data = caffe_label.tostring()

        return datum.SerializeToString()

    #
    # @brief Binarises the label. It will be thresholded so that only 0/maxval values are present.
    #
    # @param[in] thresh Values greater or equal than 'thresh' will be transformed to 'maxval'.
    # @param[in] maxval Integer that will be given to those pixels higher or equal than 'thresh'.
    #
    # @returns nothing.
    def binarise(self, thresh = 10, maxval = 1):
        _, self._raw_frame = cv2.threshold(self._raw_frame, thresh, maxval, cv2.THRESH_BINARY)

    #
    # @brief Convert intensity-based labels into proper class-index labels.
    #
    # @returns an array_like, shape (h, w).
    def to_classes(self):
        class_index_frame = self._raw_frame.copy()
        for k, v in self._class_map.items():
            class_index_frame[self._raw_frame == k] = v
        return class_index_frame

    #
    # @brief Maps between intensities [0, 255] to classes [0, K] using the JSON info provided.
    #
    # @param[in] intensity Typically an integer [0, 255].
    #
    # @returns the class index of the givel pixel intensity according to the provided class map.
    def map_intensity_to_class(self, intensity):
        return self._class_map[intensity]

    #
    # @brief Maps between classes and JSON intensities.
    #
    # @param[in] class_id Id of the class whose intensity you want to retrieve.
    #
    # @returns the intensity corresponding to the given class.
    def map_class_to_intensity(self, class_id):
        return {v: k for k, v in self._class_map.items()}[class_id]

    #
    # @brief Retrieves a normalised probability map for a particular class.
    #
    # @param[in] class_id Id of the class whose probability map you want to retrieve.
    #
    # @returns an array_like probability map, shape (h, w).
    def softmax_predicted_map(self, class_id):
        assert(self._predicted_map)
        pmap = np.exp(self._predicted_map - np.amax(self._predicted_map, axis = 0))
        pmap /= np.sum(pmap, axis = 0)

        return pmap[class_id, ...]

    #
    # @brief Converts all the feature maps to contour images.
    #
    # @param[in] pixel_width Thickness of the border in pixels.
    #
    # @returns nothing.
    def convert_to_contours(self, pixel_width = 5):

        new_raw_frame = np.zeros_like(self._raw_frame)

        # If self._predicted_map does not exist, we create it, shape (c, h, w)
        if not self._predicted_map:
            self._predicted_map = np.zeros((self._classes, self._raw_frame.shape[0], self._raw_frame.shape[1]),
                dtype=np.uint8)
            for k in range(self._classes):
                self._predicted_map[k,:,:][self._raw_frame == self.map_class_to_intensity(k)] = 1

        # Draw contours in the new raw frame
        for k in range(self._classes):
            (_, cnts, _) = cv2.findContours(self._predicted_map[k], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                # cv2.drawContours(new_raw_frame, [c], -1, (self.map_class_to_intensity(k)), pixel_width)
                cv2.drawContours(new_raw_frame, [c], -1, self.map_class_to_intensity(k), pixel_width)

        self._raw_frame = new_raw_frame

    def random_crop(self, height, width):
        self._raw_frame = CaffeinatedAbstract.random_crop(self._raw_frame, height, width)

    #
    # @brief Calculates the number of classes in the frame, that is the quantity of unique labels.
    #
    # @returns an integer that indicates the number of different pixel labels.
    @property
    def classes(self):
        return self._classes
        # return np.unique(self._raw_frame).shape[0]

    #
    # @returns the unnormalised predicted map for all the classes (class_id, height, width).
    #
    @property
    def predicted_map(self):
        return self._predicted_map

    @property
    def class_map(self):
        return self._class_map

#
# @class CaffeinatedBinaryLabel behaves as a CaffeinatedLabel but makes sure that the label images provided
#                               only contain two different types or labels. Furthermore, it makes them 0's
#                               and 1's (np.uint8) in case that they are different from these two values.
#                               Say that you provide an image with 0's and 255's as typical ground truth
#                               images, this class will make it 0's and 1's.
#
class CaffeinatedBinaryLabel(CaffeinatedLabel):
    #
    # @brief   Stores the label and checks that both, dimensions and type, are correct for a label.
    # @details If the label provided is not single channel, the label is converted to grayscale with
    #          the OpenCV cvtColour function. It
    #
    # @param[in] label_image Single channel OpenCV/Numpy image. Shape (H x W) or (H x W x C).
    # @param[in] name        String that identifies the label, usually a frame number.
    # @param[in] thresh      Values greater or equal than 'thresh' will be transformed to 'maxval'.
    # @param[in] maxval      Integer that will be given to those pixels higher or equal than 'thresh'.
    #
    # @returns nothing.
    def __init__(self, label_image, name, thresh = 10, maxval = 1):

        # Call CaffeinatedLabel constructor
        super(self.__class__, self).__init__(label_image, name)

        # Sanity check: labels that are neither (H x W) nor (H x W x C) are not supported
        # dim = len(label_image.shape)
        # if not (dim == 2 or dim == 3):
        #   raise RuntimeError('[CaffeinatedLabel] Error, the label provided has a dimension of ' + \
        #       str(dim) + ', which is not supported. Only (H x W) and (H x W x C) are supported.')

        # If we received a colour image as a label, we convert it to grayscale
        # if dim == 3:
        #   label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)

        # If the label image is not binary, that is, if it has more than two unique values, we thresholded it
        # to ensure that the labels are binary
        EXPECTED_NO_UNIQUE_VALUES = 2   # As we expect a binary label
        no_unique_values = np.unique(self._raw_frame).shape[0]
        if no_unique_values > EXPECTED_NO_UNIQUE_VALUES:
            _, self._raw_frame = cv2.threshold(self._raw_frame, thresh, maxval, cv2.THRESH_BINARY)

    #
    # @returns the number of foreground pixels.
    @property
    def count_fg_pixels(self):
        return np.count_nonzero(self._raw_frame)

    #
    # @returns the number of background pixels.
    @property
    def count_bg_pixels(self):
        return np.count_nonzero(self._raw_frame == 0)

#
# @class CaffeinatedImagePair represents a pair of consecutive frames that will be used by Caffe so this
#                             class should provide methods to adapt the original images to the type of
#                             input Caffe is expecting.
#
class CaffeinatedImagePair(object):
    #
    # @brief Saves the colour image as an attribute of the class.
    #
    # @param[in] frame_bgr_prev Numpy array with the previous BGR image in the video sequence, shape (h, w, c).
    # @param[in] frame_bgr_next Numpy array with the current BGR image in the video sequecne, shape (h, w, c).
    #
    def __init__(self, frame_bgr_prev, frame_bgr_next):
        # Sanity check: both images must have 3 dimensions (h, w, c)
        if len(frame_bgr_prev.shape) != 3 or len(frame_bgr_next.shape) != 3:
            raise RuntimeError('[CaffeinatedImagePair.__init__] The images provided must have ' + \
                'three dimensions (i.e. H x W x C).')

        # Sanity check: both images must have 3 channels
        if frame_bgr_prev.shape[2] != 3 or frame_bgr_next.shape[2] != 3:
            raise RuntimeError('[CaffeinatedImagePair.__init__] The images provided must have three ' + \
                'channels (i. e. BGR)')

        # Sanity check: both images must have the same height and width
        if frame_bgr_prev.shape[0] != frame_bgr_next.shape[0] or \
            frame_bgr_prev.shape[1] != frame_bgr_next.shape[1]:

            raise RuntimeError('[CaffeinatedImagePair.__init__] The imaged provided must have the same ' + \
                'dimensions (i.e. height and width).')

        self._frame_bgr_prev = frame_bgr_prev
        self._frame_bgr_next = frame_bgr_next

    #
    # @brief   Builds an object of type CaffeinatedImage from file.
    #
    # @details Only supports 3-channel colour images. It will raise errors for images with a different
    #          number of channels.
    #
    # @param[in] path_prev Path to the previous image file.
    # @param[in] path_next Path to the next image file.
    #
    @classmethod
    def from_file(cls, path_prev, path_next):
        return cls(cv2.imread(path_prev), cv2.imread(path_next))

    #
    # @brief   Convert image to caffe test input, transposing it to the Caffe format (C x H x W) and
    #          subtracting the training mean.
    #
    # @details The mean needs to be subtracted because there is no transform_param section in the input
    #          layer of the test network.
    #
    # @param[in] mean_values Numpy ndarray with the per channel mean of the training set. Shape (channels,).
    #
    # @returns an image ready to be processed by Caffe.
    def convert_to_caffe_input(self, mean_values):
        no_mean_values = mean_values.shape[0]

        # Sanity check: the mean values should be equal to the number of channels of the input image
        if no_mean_values != 6:
            raise ValueError('[CaffeinatedImagePair.convert_to_caffe_input()] Error, six means are required.')

        # Subtract mean values from previous frame
        norm_prev = self._frame_bgr_prev.astype(np.float32) - mean_values[:3]

        # Subtract mean values from next frame
        norm_next = self._frame_bgr_next.astype(np.float32) - mean_values[3:]

        # Sanity checks: both images must have the same shape and be of the same datatype
        assert(norm_prev.shape[0] == norm_next.shape[0])
        assert(norm_prev.shape[1] == norm_next.shape[1])
        assert(norm_prev.shape[2] == norm_next.shape[2])
        assert(norm_prev.dtype == norm_next.dtype)

        # Combine both images in a 6-channel image
        combined_image = np.empty((norm_prev.shape[0], norm_prev.shape[1], 6), dtype = norm_prev.dtype)
        combined_image[:,:, 0:3] = norm_prev
        combined_image[:,:, 3:6] = norm_next

        # Transpose to channel-first Caffe style
        combined_transposed = np.transpose(combined_image, (2, 0, 1))

        return combined_transposed

    #
    # @brief   Converts BGR image to a Caffe datum with shape (C x H x W).
    #
    # @details The training mean is not subtracted from the image because Caffe does this automatically for
    #          the data layer used for training (see the transform_param section of the 'data' layer in the
    #          training prototxt).
    #
    # @returns the Caffe datum serialised as a string.
    @property
    def serialise_to_string(self):

        # Sanity checks: both images must have the same shape and be of the same datatype
        import caffe
        assert(self._frame_bgr_prev.shape[0] == self._frame_bgr_next.shape[0])
        assert(self._frame_bgr_prev.shape[1] == self._frame_bgr_next.shape[1])
        assert(self._frame_bgr_prev.shape[2] == self._frame_bgr_next.shape[2])
        assert(self._frame_bgr_prev.dtype == self._frame_bgr_next.dtype)

        # Combine the two images in a single 6-channel image
        channels = 6
        combined_image = np.empty((self._frame_bgr_prev.shape[0], self._frame_bgr_prev.shape[1], channels), \
            dtype = self._frame_bgr_prev.dtype)
        combined_image[:,:, 0:3] = self._frame_bgr_prev
        combined_image[:,:, 3:6] = self._frame_bgr_next
        caffe_image = combined_image.astype(np.float32)

        # Convert image to Caffe datum
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height, datum.width, _ = caffe_image.shape
        datum.channels = channels
        datum.data = caffe_image.tostring()

        return datum.SerializeToString()

    #
    # @returns the height of the image.
    @property
    def height(self):
        return self._frame_bgr_prev.shape[0]

    #
    # @returns the width of the image.
    @property
    def width(self):
        return self._frame_bgr_prev.shape[1]

#
# @class CaffeinatedImagePlusPrevSeg represents a BGR image with a fourth channel that contains the segmentation of the
#                                    previous frame in the video sequence.
#
class CaffeinatedImagePrevSeg(object):
    #
    # @brief Saves the colour image and the previous segmentation as attributes of the class.
    #
    # @param[in] prev_seg  Numpy array with the predicted segmentation of the previous frame in the sequence,
    #                      shape (h, w, c).
    # @param[in] frame_bgr Numpy array with a BGR image, shape (h, w, c).
    #
    def __init__(self, prev_seg, frame_bgr):

        # Sanity check: the image must have three dimensions (h, w, c) and three channels (c = 3)
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            raise RuntimeError('[CaffeinatedImagePlusPrevSeg.__init__] Error, the image provided must ' + \
                ' have three dimensions (i.e. H x W x 3).')

        # Sanity check: the previous mask must have a dimension of two
        if len(prev_seg.shape) != 2:
            raise RuntimeError('[CaffeinatedImagePlusPrevSeg.__init__] Error, the previous mask must have ' + \
                'two dimensions.')

        # Sanity check: the frame and the previous mask must have the same dimensions
        if frame_bgr.shape[0] != prev_seg.shape[0] or frame_bgr.shape[1] != prev_seg.shape[1]:
            raise RuntimeError('[CaffeinatedImagePlusPrevSeg.__init__] Error, the current image and the ' + \
                'previous segmentation must have the same height and width.')

        self._prev_seg = prev_seg
        self._frame_bgr = frame_bgr

    #
    # @brief   Builds an object of type CaffeinatedImage from file.
    #
    # @details Only supports 3-channel colour images. It will raise errors for images with a different
    #          number of channels.
    #
    # @param[in] path Path to the image file.
    @classmethod
    def from_file(cls, path_prev_seg, path_frame_bgr):
        caffeinated_prev_label = CaffeinatedBinaryLabel.from_file(path_prev_seg)

        return cls(caffeinated_prev_label.single_channel_label_copy(), cv2.imread(path_frame_bgr))

    #
    # @brief   Convert image to caffe test input, transposing it to the Caffe format (C x H x W) and
    #          subtracting the training mean.
    #
    # @details The mean needs to be subtracted because there is no transform_param section in the input
    #          layer of the test network.
    #
    # @param[in] mean_values Numpy ndarray with the per channel mean of the training set. Shape (channels,).
    #
    # @returns an image ready to be processed by Caffe.
    def convert_to_caffe_input(self, mean_values):
        colour_channels = 3
        no_mean_values = mean_values.shape[0]

        # Sanity check: the mean values should be equal to the number of channels of the input image
        if no_mean_values != colour_channels:
            raise ValueError('[CaffeinatedImagePlusPrevSeg.convert_to_caffe_input] Error, three means are ' + \
                'required.')

        # Subtract mean values from the current frame
        norm_frame_bgr = self._frame_bgr.astype(np.float32) - mean_values

        # Convert previous segmentation to float
        norm_prev_seg = self._prev_seg.astype(np.float32)

        # Sanity check: the current normalised image and the segmentation mask must have the same shape and
        #               datatype
        total_channels = colour_channels + 1
        assert(norm_frame_bgr.shape[0] == norm_prev_seg.shape[0])
        assert(norm_frame_bgr.shape[1] == norm_prev_seg.shape[1])
        assert(norm_frame_bgr.shape[2] == colour_channels)
        assert(norm_frame_bgr.dtype == norm_prev_seg.dtype)

        # Combine the current frame with the previous segmentation in a 4-channel image
        combined_image = np.empty((norm_frame_bgr.shape[0], norm_frame_bgr.shape[1], total_channels),
            dtype = norm_frame_bgr.dtype)
        combined_image[:,:, :colour_channels] = norm_frame_bgr
        combined_image[:,:, colour_channels] = norm_prev_seg

        # Transpose to channel-first Caffe style
        combined_transposed = np.transpose(combined_image, (2, 0, 1))

        return combined_transposed

    #
    # @brief   Converts BGR image to a Caffe datum with shape (C x H x W).
    #
    # @details The training mean is not subtracted from the image because Caffe does this automatically for
    #          the data layer used for training (see the transform_param section of the 'data' layer in the
    #          training prototxt).
    #
    # @returns the Caffe datum serialised as a string.
    @property
    def serialise_to_string(self):

        # Sanity checks: both images must have the same shape and be of the same datatype
        import caffe
        assert(self._frame_bgr.shape[0] == self._prev_seg.shape[0])
        assert(self._frame_bgr.shape[1] == self._prev_seg.shape[1])
        assert(self._frame_bgr.dtype == self._prev_seg.dtype)

        # Combine the current image and the previous segmentation in a single 4-channel image
        colour_channels = 3
        total_channels = colour_channels + 1
        combined_image = np.empty((self._frame_bgr.shape[0], self._frame_bgr.shape[1], total_channels), \
            dtype = self._frame_bgr.dtype)
        combined_image[:,:, :colour_channels] = self._frame_bgr
        combined_image[:,:, colour_channels] = self._prev_seg
        caffe_image = combined_image.astype(np.float32)

        # Convert image to Caffe datum
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height, datum.width, _ = caffe_image.shape
        datum.channels = total_channels
        datum.data = caffe_image.tostring()

        return datum.SerializeToString()

    #
    # @returns the height of the image.
    @property
    def height(self):
        return self._frame_bgr.shape[0]

    #
    # @returns the width of the image.
    @property
    def width(self):
        return self._frame_bgr.shape[1]

#
# @brief Convert a binary probability map into a beautiful image.
#
# @param[in] probmap 2D floating point probability map, shape (height, width).
#
# @returns a fancy BGR image.
def make_it_pretty(probmap, vmin = 0, vmax = 1, colourmap = 'plasma', eps = 1e-3):
    assert(len(probmap.shape) == 2)
    assert(np.max(probmap) < vmax + eps)
    assert(np.min(probmap) > vmin - eps)
    height = probmap.shape[0]
    width = probmap.shape[1]

    # Create figure without axes
    fig = plt.figure(frameon = False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Plot figure
    plt.imshow(probmap, cmap = colourmap, vmin = vmin, vmax = vmax) # vmin/vmax adjust thesholds
    fig.canvas.draw()

    # Convert plot to numpy array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    # Remove left/right borders
    left_right_offset = 0
    i = 0
    left_intensity = data[0, left_right_offset, 0]
    right_intensity = data[0, -1, 0]
    min_intensity = 255

    # Assert that the values for all the rows are equal for the columns 'offset' and '-offset'
    left_side_equal = True if np.unique(data[:, left_right_offset, 0]).shape[0] == 1 else False
    right_side_equal = True if np.unique(data[:, -left_right_offset, 0]).shape[0] == 1 else False

    while left_intensity == right_intensity and left_intensity >= min_intensity and left_side_equal and right_side_equal:
        left_right_offset += 1
        left_intensity = data[0, left_right_offset, 0]
        right_intensity = data[0, -left_right_offset - 1, 0]
        left_side_equal = True if np.unique(data[:, left_right_offset, 0]).shape[0] == 1 else False
        right_side_equal = True if np.unique(data[:, -left_right_offset, 0]).shape[0] == 1 else False

    # Remove top/bottom borders
    top_bottom_offset = 0
    i = 0
    top_intensity = data[top_bottom_offset, 0, 0]
    bottom_intensity = data[-1, 0, 0]
    min_intensity = 255

    # Assert that the values for all the rows are equal for the columns 'offset' and '-offset'
    top_side_equal = True if np.unique(data[top_bottom_offset,:, 0]).shape[0] == 1 else False
    bottom_side_equal = True if np.unique(data[-top_bottom_offset,:, 0]).shape[0] == 1 else False

    while top_intensity == bottom_intensity and top_intensity >= min_intensity and top_side_equal and bottom_side_equal:
        top_bottom_offset += 1
        top_intensity = data[top_bottom_offset, 0, 0]
        bottom_intensity = data[-top_bottom_offset - 1, 0, 0]
        top_side_equal = True if np.unique(data[top_bottom_offset,:, 0]).shape[0] == 1 else False
        bottom_side_equal = True if np.unique(data[-top_bottom_offset,:,  0]).shape[0] == 1 else False

    # Note: 1 is added to 'left_right_offset' because matplotlib tends to leave a border on the left one
    # pixel thicker than on the right
    cropped_image = data[top_bottom_offset:data.shape[0] - top_bottom_offset,
        left_right_offset + 1:data.shape[1] - left_right_offset]

    # Resize to original size
    resized_image = cv2.resize(cropped_image, (width, height))
    assert(resized_image.shape[0] == height)
    assert(resized_image.shape[1] == width)
    assert(resized_image.shape[2] == 3)

    # Convert RGB to BGR
    final_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    return final_image

# This module cannot be executed as a script because it is not a script :)
if __name__ == "__main__":
    print >> sys.stderr, 'Error, this module is not supposed to be executed by itself.'
    sys.exit(1)
