"""
@brief   This module generates synthetic images for tool segmentation.
@author  Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
@date    11 Mar 2019.
"""

# import random
import os
import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator as KerasGenerator
import albumentations
import scipy.ndimage
import scipy.ndimage.interpolation
import skimage.morphology
import random
import noise.perlin

# My imports
import common
import image
import blending
import geometry

class ToolCC:
    """
    @class ToolCC represents a tool connected component where the edge pixels are identified.
    """
    def __init__(self, im, mask, ep_mask):
        """
        @param[in]  im       Original image, the same rotation will be performed in both image,
                             mask, and entrypoints.
        @param[in]  mask     Binary (0/1) mask with positive pixels indicating tool presence.
        @param[in]  ep_mask  Binary mask with positive pixels indicating the border of the tool.
                             This is expected to be a subset of the 'mask'.
        """
        self.im = im
        self.mask = mask
        self.ep_mask = ep_mask

    def rotate(self, deg):
        """
        @brief Rotate the tool mask along with the entrypoints.

        @param[in] deg Angle of rotation in degrees.

        @returns nothing.
        """

        if deg == 0:
            return

        # We get the original shape to make sure the rotated one has the same shape
        prev_shape = self.im.shape

        # Find centre of mass of the tool
        cm_y, cm_x = scipy.ndimage.center_of_mass(self.mask)
        cm_y = int(round(cm_y))
        cm_x = int(round(cm_x))

        # Rotate tool and mask around the centre of mass
        im_rot = image.CaffeinatedAbstract.rotate_bound_centre(self.im,
            (cm_x, cm_y), deg, cv2.INTER_LANCZOS4)
        mask_rot = image.CaffeinatedAbstract.rotate_bound_centre(self.mask,
            (cm_x, cm_y), deg, cv2.INTER_NEAREST)

        # Find out whether we are dealing with a corner case or not
        sides = geometry.entry_sides_in_mask(self.mask)
        single_side = False
        if len(sides) == 1:
            single_side = True

        # Crop depending on whether it is a single side or a corner case
        if single_side:
            ep_mask_rot = image.CaffeinatedAbstract.rotate_bound_centre(
                self.ep_mask, (cm_x, cm_y), deg, cv2.INTER_NEAREST)

            # Rotate tool that is connected to only one side of the image
            self.im, self.mask, self.ep_mask = self.crop_rotated_single_border_case(im_rot,
                mask_rot, ep_mask_rot)
        elif deg == 180: # This is the only rotation allowed for complex tool configurations
            self.im = im_rot
            self.mask = mask_rot
            self.ep_mask = image.CaffeinatedAbstract.rotate_bound_centre(
                self.ep_mask, (cm_x, cm_y), deg, cv2.INTER_NEAREST)
        elif sides == set(['left', 'top']) \
            or sides == set(['top', 'right']) \
            or sides == set(['right', 'bottom']) \
            or sides == set(['bottom', 'left']):
            # The tool is touching a corner of the image: two crops are needed to re-attach it to
            # the border of the image after the random rotation

            # Rotate keypoints
            p1, p2, p3 = ToolCC.get_corner_keypoints(self.ep_mask, dilate_ksize=3)
            rot_mat = ToolCC.get_rotate_bound_centre_matrix(
                self.mask.shape[0], self.mask.shape[1], (cm_x, cm_y),
                deg, cv2.INTER_NEAREST)
            p1 = np.round(np.dot(rot_mat, p1)).astype(np.int)
            p2 = np.round(np.dot(rot_mat, p2)).astype(np.int)
            p3 = np.round(np.dot(rot_mat, p3)).astype(np.int)
            p1_x = p1[0, 0]
            p1_y = p1[1, 0]
            p2_x = p2[0, 0]
            p2_y = p2[1, 0]
            p3_x = p3[0, 0]
            p3_y = p3[1, 0]

            # Define cropping points to leave rotated tool connected to the borders
            y_crop_start = 0
            y_crop_end = mask_rot.shape[0]
            x_crop_start = 0
            x_crop_end = mask_rot.shape[1]
            min_x = np.min(np.where(mask_rot)[1])
            max_x = np.max(np.where(mask_rot)[1])
            min_y = np.min(np.where(mask_rot)[0])
            max_y = np.max(np.where(mask_rot)[0])
            tolerance = 20 # pixels
            if p3_y < p1_y and p3_y < p2_y:     # /\
                if np.abs(p2_y - max_y) < tolerance:
                    y_crop_start = p1_y
                elif np.abs(p1_y - max_y) < tolerance:
                    y_crop_start = p2_y
                else:
                    y_crop_start = max(p1_y, p2_y)
            elif p3_y > p1_y and p3_y > p2_y:   # \/
                if np.abs(p2_y - min_y) < tolerance:
                    y_crop_end = p1_y
                elif np.abs(p1_y - min_y) < tolerance:
                    y_crop_end = p2_y
                else:
                    y_crop_end = min(p1_y, p2_y)
            elif p3_x < p1_x and p3_x < p2_x:   # <
                if np.abs(p2_x - max_x) < tolerance:
                    x_crop_start = p1_x
                elif np.abs(p1_x - max_x) < tolerance:
                    x_crop_start = p2_x
                else:
                    x_crop_start = max(p1_x, p2_x)
            elif p3_x > p1_x and p3_x > p2_x:   # >
                if np.abs(p2_x - min_x) < tolerance:
                    x_crop_end = p1_x
                elif np.abs(p1_x - min_x) < tolerance:
                    x_crop_end = p2_x
                else:
                    x_crop_end = min(p1_x, p2_x)

            # Crop image and mask, create new ep_mask according to the crop
            im = im_rot[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
            mask = mask_rot[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
            ep_mask = np.zeros_like(self.mask)
            ep_mask[0,:] = self.mask[0,:]
            ep_mask[-1,:] = self.mask[-1,:]
            ep_mask[:, 0] = self.mask[:, 0]
            ep_mask[:, -1] = self.mask[:, -1]

            # Pad or crop accordingly to come back to the original image size
            new_sides = geometry.entry_sides_in_mask(mask)
            self.im, self.mask, self.ep_mask = self.adjust_rotated_image(im,
                mask, ep_mask, new_sides.pop())
        else:
            common.writeln_warn('This tool configuation cannot be rotated.')

        # Sanity check: rotation should not change the dimensions of the
        #               image
        assert(self.im.shape[0] == prev_shape[0])
        assert(self.im.shape[1] == prev_shape[1])

    @staticmethod
    def get_rotate_bound_centre_matrix(h, w, centre, deg, interp):
        cm_x = centre[0]
        cm_y = centre[1]

        # Build the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((cm_y, cm_x), -deg, 1.0)
        rot_mat_hom = np.zeros((3, 3))
        rot_mat_hom[:2,:] = rot_mat
        rot_mat_hom[2, 2] = 1

        # Find the coordinates of the corners in the rotated image
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

        # Create homogeneous rotation matrix
        hom_rot_mat = np.zeros((3, 3), dtype=np.float64)
        hom_rot_mat[0:2] = rot_mat
        hom_rot_mat[2, 2] = 1

        return hom_rot_mat

    @staticmethod
    def get_corner_keypoints(ep_mask, dilate_ksize=0):
        """
        @brief It gets a mask of entrypoints of a tool attached to a corner
               and produces a mask with three points, labelling 1 the
               shortest side of the triangle, 2 the longest, and 3 the corner.
        @param[in] ep_mask Binary mask with those pixels != 0 representing
                           the points of the tool that are touching the
                           borders of the image.
        @returns three points p1, p2, p3 in homogeneous [x, y, 1]
                 coordinates.
        """

        h = ep_mask.shape[0]
        w = ep_mask.shape[1]
        min_x = np.min(np.where(ep_mask)[1])
        max_x = np.max(np.where(ep_mask)[1])
        min_y = np.min(np.where(ep_mask)[0])
        max_y = np.max(np.where(ep_mask)[0])

        # Amend mask in case of 1-pixel stupid gaps
        amended_mask = None
        if dilate_ksize > 2:
            kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
            amended_mask = cv2.dilate(ep_mask, kernel, iterations = 1)
            if max_x < amended_mask.shape[1] - 1:
                amended_mask[:, max_x + 1:] = 0
            if min_x > 0:
                amended_mask[:, :min_x] = 0
            if max_y < amended_mask.shape[0] - 1:
                amended_mask[max_y + 1:,:] = 0
            if max_y > 0:
                amended_mask[:min_y,:] = 0
        else:
            amended_mask = ep_mask

        kp_mask = np.zeros_like(amended_mask)
        if amended_mask[0, 0] != 0: # Top left corner
            kp_mask[0, 0] = 3
            if max_x < max_y:
                kp_mask[0, max_x] = 1
                kp_mask[max_y, 0] = 2
            else:
                kp_mask[0, max_x] = 2
                kp_mask[max_y, 0] = 1
        elif amended_mask[0, w - 1] != 0: # Top right corner
            kp_mask[0, w - 1] = 3
            if w - min_x < max_y:
                kp_mask[0, min_x] = 1
                kp_mask[max_y, w - 1] = 2
            else:
                kp_mask[0, min_x] = 2
                kp_mask[max_y, w - 1] = 1
        elif amended_mask[h - 1, 0] != 0: # Bottom left corner
            kp_mask[h - 1, 0] = 3
            if h - min_y < max_x:
                kp_mask[min_y, 0] = 1
                kp_mask[h - 1, max_x] = 2
            else:
                kp_mask[min_y, 0] = 2
                kp_mask[h - 1, max_x] = 1
        elif amended_mask[h - 1, w - 1] != 0: # Bottom right corner
            kp_mask[h - 1, w - 1] = 3
            if h - min_y < w - min_x:
                kp_mask[min_y, w - 1] = 1
                kp_mask[h - 1, min_x] = 2
            else:
                kp_mask[min_y, w - 1] = 2
                kp_mask[h - 1, min_x] = 1

        # Get point coordinates in format [x y 1].T
        p1 = np.array([0, 0, 1]).reshape((3, 1))
        p2 = np.array([0, 0, 1]).reshape((3, 1))
        p3 = np.array([0, 0, 1]).reshape((3, 1))
        p1[0:2] = np.flipud(np.array(np.where(kp_mask == 1)))
        p2[0:2] = np.flipud(np.array(np.where(kp_mask == 2)))
        p3[0:2] = np.flipud(np.array(np.where(kp_mask == 3)))

        return p1, p2, p3


    def crop_rotated_single_border_case(self, im_rot, mask_rot, ep_mask_rot):
        """
        @brief Crop a rotated tool so that it can be reattached to the border of the image.
        @details This function only deals with the case where the tool to be rotated touches only
                 one corner.
        """

        # Find the positions of the min/max x/y coordinates of the entrypoints in the rotated image
        min_x = np.min(np.where(ep_mask_rot)[1])
        max_x = np.max(np.where(ep_mask_rot)[1])
        min_y = np.min(np.where(ep_mask_rot)[0])
        max_y = np.max(np.where(ep_mask_rot)[0])

        # Compute the four possible crops to keep the tool attached to the
        # border
        min_x_image = im_rot[:, :min_x]
        max_x_image = im_rot[:, max_x:]
        min_y_image = im_rot[:min_y,:]
        max_y_image = im_rot[max_y:,:]
        min_x_mask = mask_rot[:, :min_x]
        max_x_mask = mask_rot[:, max_x:]
        min_y_mask = mask_rot[:min_y,:]
        max_y_mask = mask_rot[max_y:,:]

        # Compute the amount of tool pixels that each crop leaves inside the image
        images = [min_x_image, max_x_image, min_y_image, max_y_image]
        masks = [min_x_mask, max_x_mask, min_y_mask, max_y_mask]
        sides = ['right', 'left', 'bottom', 'top']
        pixels = [np.nonzero(mask)[0].shape[0] for mask in masks]

        # Keep the crop that leaves more tool inside the image
        best_idx = np.argmax(pixels)
        best_im = images[best_idx]
        best_mask = masks[best_idx]
        border_side = sides[best_idx]

        # Create new mask of entrypoints based on the rotated and cropped image
        best_ep_mask = np.zeros_like(best_mask, dtype=np.uint8)
        if border_side == 'right':
            best_ep_mask[:, -1] = 1
        elif border_side == 'left':
            best_ep_mask[:, 0] = 1
        elif border_side == 'bottom':
            best_ep_mask[-1,:] = 1
        elif border_side == 'top':
            best_ep_mask[0,:] = 1
        best_ep_mask *= best_mask

        # Pad or crop accordingly to come back to the original image size
        new_im, new_mask, new_ep_mask = self.adjust_rotated_image(best_im,
            best_mask, best_ep_mask, border_side)

        return new_im, new_mask, new_ep_mask

    def adjust_rotated_image(self, im, mask, ep_mask, border_side):
        im = im.copy()
        mask = mask.copy()
        ep_mask = ep_mask.copy()

        # Compute the coordinates of the tool within the new image
        new_tl_y = np.min(np.where(mask)[0])
        new_tl_x = np.min(np.where(mask)[1])
        new_br_y = np.max(np.where(mask)[0])
        new_br_x = np.max(np.where(mask)[1])
        new_tool_height = new_br_y + 1 - new_tl_y
        new_tool_width = new_br_x + 1 - new_tl_x

        # Compute the proportions of vertical free space in the new image
        new_top_free_space = new_tl_y
        new_bottom_free_space = im.shape[0] - new_br_y
        new_vertical_free_space = new_tl_y + (im.shape[0] - new_br_y)
        new_top_free_space_prop = float(new_top_free_space) / new_vertical_free_space
        new_bottom_free_space_prop = float(new_bottom_free_space) / new_vertical_free_space

        # Compute the proportions of horizontal free space in the new image
        new_left_free_space = new_tl_x
        new_right_free_space = im.shape[1] - new_br_x
        new_horizontal_free_space = new_left_free_space + new_right_free_space
        new_left_free_space_prop = float(new_left_free_space) / new_horizontal_free_space
        new_right_free_space_prop = float(new_right_free_space) / new_horizontal_free_space

        if mask.shape[0] > self.mask.shape[0]: # We have to cut height
            if border_side == 'top':
                cut = self.im.shape[0]
                im = im[:cut,:]
                mask = mask[:cut,:]
                ep_mask = ep_mask[:cut,:]
            elif border_side == 'bottom':
                cut = im.shape[0] - self.im.shape[0]
                im = im[cut:,:]
                mask = mask[cut:,:]
                ep_mask = ep_mask[cut:,:]
            else: # border is left or right, we cut from top and bottom
                top_cut = int(round(new_tl_y - (self.im.shape[0] - new_tool_height) * new_top_free_space_prop))
                bottom_cut = top_cut + self.im.shape[0]
                im = im[top_cut:bottom_cut,:]
                mask = mask[top_cut:bottom_cut,:]
                ep_mask = ep_mask[top_cut:bottom_cut,:]
        else: # We have to pad height
            if border_side == 'top':
                bpad = self.im.shape[0] - im.shape[0]
                im = np.pad(im, ((0, bpad), (0, 0), (0, 0)), 'constant', constant_values=(0))
                mask = np.pad(mask, ((0, bpad), (0, 0)), 'constant', constant_values=(0))
                ep_mask = np.pad(ep_mask, ((0, bpad), (0, 0)), 'constant', constant_values=(0))
            elif border_side == 'bottom':
                tpad = self.im.shape[0] - im.shape[0]
                im = np.pad(im, ((tpad, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                mask = np.pad(mask, ((tpad, 0), (0, 0)), 'constant', constant_values=(0))
                ep_mask = np.pad(ep_mask, ((tpad, 0), (0, 0)), 'constant', constant_values=(0))
            else: # border is left or right
                extra = self.im.shape[0] - im.shape[0]
                tpad = int(round(extra * new_top_free_space_prop))
                bpad = extra - tpad
                im = np.pad(im, ((tpad, bpad), (0, 0), (0, 0)), 'constant', constant_values=(0))
                mask = np.pad(mask, ((tpad, bpad), (0, 0)), 'constant', constant_values=(0))
                ep_mask = np.pad(ep_mask, ((tpad, bpad), (0, 0)), 'constant', constant_values=(0))

        if mask.shape[1] > self.mask.shape[1]: # We have to cut width
            if border_side == 'left':
                cut = self.im.shape[1]
                im = im[:, :cut]
                mask = mask[:, :cut]
                ep_mask = ep_mask[:, :cut]
            elif border_side == 'right':
                cut = im.shape[1] - self.im.shape[1]
                im = im[:, cut:]
                mask = mask[:, cut:]
                ep_mask = ep_mask[:, cut:]
            else: # border is top or bottom, we cut from left and right
                left_cut = int(round(new_tl_x - (self.im.shape[1] - new_tool_width) \
                    * new_left_free_space_prop))
                right_cut = left_cut + self.im.shape[1]
                im = im[:, left_cut:right_cut]
                mask = mask[:, left_cut:right_cut]
                ep_mask = ep_mask[:, left_cut:right_cut]
        else: # We have to pad width
            if border_side == 'left':
                rpad = self.im.shape[1] - im.shape[1]
                im = np.pad(im, ((0, 0), (0, rpad), (0, 0)), 'constant', constant_values=(0))
                mask = np.pad(mask, ((0, 0), (0, rpad)), 'constant', constant_values=(0))
                ep_mask = np.pad(ep_mask, ((0, 0), (0, rpad)), 'constant', constant_values=(0))
            elif border_side == 'right':
                lpad = self.im.shape[1] - im.shape[1]
                im = np.pad(im, ((0, 0), (lpad, 0), (0, 0)), 'constant', constant_values=(0))
                mask = np.pad(mask, ((0, 0), (lpad, 0)), 'constant', constant_values=(0))
                ep_mask = np.pad(ep_mask, ((0, 0), (lpad, 0)), 'constant', constant_values=(0))
            else: # We have to pad left and right
                extra = self.im.shape[1] - im.shape[1]
                lpad = int(round(extra * new_left_free_space_prop))
                rpad = extra - lpad
                im = np.pad(im, ((0, 0), (lpad, rpad), (0, 0)), 'constant', constant_values=(0))
                mask = np.pad(mask, ((0, 0), (lpad, rpad)), 'constant', constant_values=(0))
                ep_mask = np.pad(ep_mask, ((0, 0), (lpad, rpad)), 'constant', constant_values=(0))

        return im, mask, ep_mask

class BloodDroplet:
    def __init__(self, contour, height, width, min_hsv_hue=0, max_hsv_hue=10, min_hsv_sat=50,
                 max_hsv_sat=255, max_hsv_val=200):
        """
        @param[in]  contour  Array of floating 2D points in image coordinates.
        @param[in]  height   Height of the image with the blood droplet.
        @param[in]  width    Width of the image with the blood droplet.
        """
        contour = np.round(contour).astype(np.int)

        # Correct those pixels outside the image plane
        contour[contour < 0] = 0
        contour[:, 0][contour[:, 0] > width - 1] = width - 1
        contour[:, 1][contour[:, 1] > height - 1] = height - 1

        # Generate segmentation mask for the blood droplet
        self.seg = np.zeros((height, width), dtype=np.uint8)
        self.seg[contour[:, 1], contour[:, 0]] = 255

        # Get a point inside the contour to use it as filling seed: we use the centroid
        cx = np.round(np.mean(contour[:, 0])).astype(np.int)
        cy = np.round(np.mean(contour[:, 1])).astype(np.int)

        # Fill the wholes of the blood droplet contour
        cv2.floodFill(self.seg, None, (cx, cy), 255)

        # Generate an empty image of the blood sample
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Initialise HSV image of the blood droplet
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.frame[:,:, 0] = np.random.randint(min_hsv_hue, max_hsv_hue)
        self.frame[:,:, 1] = np.random.randint(min_hsv_sat, max_hsv_sat + 1)

        min_x = np.min(np.where(self.seg)[1])
        max_x = np.max(np.where(self.seg)[1])
        min_y = np.min(np.where(self.seg)[0])
        max_y = np.max(np.where(self.seg)[0])
        drop_h = max_y + 1 - min_y
        drop_w = max_x + 1 - min_x

        # Generate Perlin noise for the V channel of the HSV image of the blood droplet
        # min_scale = 1
        # max_scale = 5
        # scale = np.random.randint(min_scale, max_scale + 1)
        scale = 1
        noise = image.perlin2d_smooth(drop_h, drop_w, scale)

        # Set the V channel to the Perlin noise
        self.frame[min_y:max_y + 1, min_x:max_x + 1, 2] = \
            np.round(noise * max_hsv_val).astype(np.uint8)

        # Convert image back to BGR and zero those pixels that are not within the droplet mask
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_HSV2BGR)
        self.frame[self.seg == 0] = 0


    #@staticmethod
    # def circle(r, n=100):
    #    """
    #    @returns a list of lists of (x, y) point coordinates.
    #    """
    #    return [(np.cos(2 * np.pi / n * x) * r, np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]


    @staticmethod
    def blob(delta=0.01, min_sf=0.1, max_sf=0.5):
        """
        @brief This method generates a list of points representing a slightly deformed
               circumference.

        @param[in]  delta           Spacing between circle points. Smaller means
                                    more circumference points will be generated.
        @param[in]  min_sf          Minimum scaling factor applied to the noise. A smaller
                                    value will produce blobs closer to a circle. Higher values will
                                    make it closer to a flower.
        @param[in]  max_sf          Maximum scaling factor.

        @returns an array of [x, y] points.
        """
        noise_gen = noise.perlin.SimplexNoise()
        noise_gen.randomize()
        points = []
        scaling_factor = np.random.uniform(min_sf, max_sf)
        for a in np.arange(0, 2 * np.pi, delta).tolist():
            xoff = np.cos(a)
            yoff = np.sin(a)
            r = noise_gen.noise2(scaling_factor * xoff, scaling_factor * yoff) + 1.
            x = r * xoff
            y = r * yoff
            points.append([x, y])

        # Normalise contour to zero mean and one std
        contour = np.array(points)
        contour -= np.mean(contour, axis=0)
        contour /= np.std(contour, axis=0)

        return contour


    @classmethod
    def from_circle(cls, cx, cy, radius, height, width):
        # Generate the droplet geometry using Perlin noise
        drop_geo = np.array(BloodDroplet.blob())

        # Transform the blob to have the desired radius
        drop_geo *= radius

        # Transform the blob to the desired location
        drop_geo[:, 0] += cx
        drop_geo[:, 1] += cy

        return cls(drop_geo, height, width)

class CustomBackend:
    def __init__(self, custom_dic, rs=np.random.RandomState(None)):
        self.custom_dic = custom_dic
        self.rs = rs


    def augment(self, raw_image, raw_label=None):
        """
        @brief Each augmentation in the internal dictionary should have a method in this class.

        @param[in]  raw_image  OpenCV/Numpy ndarray containing a BGR image.
        @param[in]  raw_label  OpenCV/Numpy ndarray of shape (height, width) containing the
                               segmentation label.

        @returns a pair of image, label. Both are of type numpy ndarray.
        """
        assert(isinstance(raw_image, np.ndarray))

        new_image = raw_image
        new_label = raw_label

        if raw_label is not None:
            assert(isinstance(raw_label, np.ndarray))

        # Randomise the order of the augmentation effects
        keys = np.array(self.custom_dic.keys())
        # np.random.shuffle(keys)
        keys = keys.tolist()

        # Apply all the augmentations that can be applied by this engine
        for aug_method in keys:
            if aug_method in AugmentationEngine.BACKENDS['custom']:
                new_image, new_label = getattr(self, aug_method)(self.custom_dic[aug_method],
                    new_image, new_label)
            else:
                common.writeln_warn(aug_method + ' is unknown to the CustomBackend.')

        return new_image, new_label


    def tool_gray(self, param, raw_image, raw_label=None):
        """
        @brief Converts the image to grayscale adding a tiny bit of uniform noise.

        @param[in]  param      Either zero (does nothing) or one (converts to grayscale).
        @param[in]  raw_image  Numpy ndarray, shape (height, width, 3).
        @param[in]  raw_label  Numpy ndarray, shape (height, width).

        @returns a pair of image, label. Shapes are like input shapes.
        """
        assert(isinstance(raw_image, np.ndarray))
        if raw_label is not None:
            assert(isinstance(raw_label, np.ndarray))

        new_image = None
        new_label = raw_label # This method does not touch the segmentation mask

        # Convert tools to grayscale with a bit of noise
        if param == 0 or param is False:
            # Don't do anything
            new_image = raw_image
        elif param == 1 or param is True:
            new_image = image.CaffeinatedImage.gray_tools(raw_image)
        else:
            raise ValueError('Error, gray_tools() does not understand the parameter ' + str(param))

        return new_image, new_label

    def tool_rotation(self, rot_range_deg, raw_image, raw_label, margin=1, dilate_ksize=3):
        """
        @brief Performs an independent random rotation on each individual tool in the image.

        @param[in]  rot_range_deg  Range of rotation, e.g. 45 would mean from -45 to +45 degrees.

        @returns the pair image, label both with the same rotation applied.
        """
        assert(rot_range_deg >= 0 and rot_range_deg <= 360)
        if raw_label is None:
            raise ValueError('tool_rotate is an augmentation that only works if there is a label.')

        # Get a random angle of rotation
        ang = np.random.randint(-rot_range_deg, rot_range_deg + 1)

        # Convert mask to 0/1
        label = np.zeros_like(raw_label, dtype=np.uint8)
        label[raw_label != 0] = 1

        # Detect entrypoints
        ep_mask = geometry.entry_points_in_mask(label, margin, dilate_ksize).astype(np.uint8)

        # If there is more than one insertion point (i.e. more than one connected component
        # in the insertion mask)
        if np.amax(ep_mask) > 1:
            # If the tool has several insertion points only rotations of [0, 90, 180, 270]
            # degrees can be performed while maintaining a realistic appearance for the tool
            ang = np.random.choice(np.arange(0, rot_range_deg * 2, 180))

        # Turn mask back into a 0/255 mask
        ep_mask[ep_mask == 1] = 255

        # Create ToolCC object
        tool = ToolCC(raw_image, raw_label, ep_mask)

        # Rotate tool
        tool.rotate(ang)

        return tool.im, tool.mask


    def tool_shift(self, param, raw_image, raw_label):
        """
        @param[in] param is not used in this augmentation method.
        """

        # Find out which sides are being touched by the mask
        sides = geometry.entry_sides_in_mask(raw_label)

        # if len(sides) > 2:
        #    common.writeln_warn('The image provided cannot be shifted, it has ' \
        #        + 'more than two points of contact with the borders.')
        new_image = raw_image
        new_label = raw_label

        # Horizontal shift
        if sides == set(['top']) or sides == set(['bottom']) or sides == set(['top', 'bottom']):
            if common.randbin():
                new_image, new_label = CustomBackend.shift_left(raw_image, raw_label)
            else:
                new_image, new_label = CustomBackend.shift_right(raw_image, raw_label)
        else:
            common.writeln_warn('The image provided cannot be shifted horizontally.')

        # Vertical shift
        if sides == set(['left']) or sides == set(['right']) or sides == set(['left', 'right']):
            if common.randbin():
                new_image, new_label = CustomBackend.shift_top(raw_image, raw_label)
            else:
                new_image, new_label = CustomBackend.shift_bottom(raw_image, raw_label)
        else:
            common.writeln_warn('The image provided cannot be shifted vertically.')

        # TODO: it is possible to shift corner cases, but they are not considered in this code

        return new_image, new_label


    def blend_border(self, param, raw_image, raw_label, max_pad=50, max_std=10, noise_p=0.5, rect_p=0.5):
        """
        @param[in] param   Probability of adding a border to the image.
        @param[in] max_pad Maximum border padding that can be added to then image.
        """
        black_im = raw_image
        black_label = raw_label

        # Add border to the image
        h = raw_image.shape[0]
        w = raw_image.shape[1]
        proba = self.rs.binomial(1, param)
        if proba:

            # Compute random padding while respecting the form factor of the image
            top_pad = self.rs.randint(1, max_pad + 1)
            bottom_pad = top_pad
            left_pad = int(round(0.5 * (((w * (h + 2 * top_pad)) / h) - w)))
            right_pad = left_pad

            # Create the black background
            black_im = np.zeros((h + top_pad + bottom_pad, w + left_pad + right_pad, 3), dtype=np.uint8)
            black_label = np.zeros((h + top_pad + bottom_pad, w + left_pad + right_pad), dtype=np.uint8)

            # Add Gaussian noise to the border
            if self.rs.binomial(1, noise_p):
                size = black_im.shape[0] * black_im.shape[1] * 3
                random_std = self.rs.randint(1, max_std)
                noise = self.rs.normal(0, random_std, size).reshape((black_im.shape[0], black_im.shape[1], 3))
                black_im = np.round(np.clip(black_im + noise, 0, max_std)).astype(np.uint8)

            # Cropping from the original image to the new one
            if self.rs.binomial(1, rect_p):
                # Rectangular crop
                black_im[top_pad:-bottom_pad, left_pad:-right_pad] = raw_image
                black_label[top_pad:-bottom_pad, left_pad:-right_pad] = raw_label
            else:
                # Circular crop
                half_h = h // 2
                half_w = w // 2
                radius = self.rs.randint(min(half_h, half_w), max(half_h, half_w) + 1)
                color = 255
                prev_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(prev_mask, (prev_mask.shape[1] // 2, prev_mask.shape[0] // 2), radius, color, -1)
                next_mask = np.pad(prev_mask, ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant', constant_values=(0))
                black_im[next_mask == color] = raw_image[prev_mask == color]
                black_label[next_mask == color] = raw_label[prev_mask == color]

        return black_im, black_label


    def blood_droplets(self, param, raw_image, raw_label, min_ndrop=5, max_ndrop=10,
                       min_radius=1, max_radius=32, blending_mode='gaussian'):
        new_image = raw_image
        new_label = raw_label

        # We add blood droplets following the probability given in the command line
        proba = self.rs.binomial(1, param)
        if proba:
            # Get those pixel locations belonging to the tool, so that we choose randomly and place
            # droplets inside the tool
            tool_pixels = np.nonzero(new_label)

            # Generate and blend blood droplets onto the image
            ndrop = self.rs.randint(min_ndrop, max_ndrop + 1)
            height = new_image.shape[0]
            width = new_image.shape[1]
            # droplets = []
            for i in range(ndrop):

                # Generate droplet
                chosen_centre = self.rs.randint(tool_pixels[0].shape[0])
                cx = tool_pixels[1][chosen_centre]
                cy = tool_pixels[0][chosen_centre]
                radius = self.rs.randint(min_radius, max_radius)
                droplet = BloodDroplet.from_circle(cx, cy, radius, height, width)

                # Blend droplet with Gaussian blending
                image_cx = int(round(.5 * new_image.shape[1]))
                image_cy = int(round(.5 * new_image.shape[0]))
                new_image = blending.blend(droplet.frame, droplet.seg, new_image.copy(), image_cy,
                                           image_cx, 'gaussian')

            # Generate Perlin noise image
            # scale = np.random.randint(min_scale, max_scale + 1)
            # perlin = image.perlin2d_smooth(raw_image.shape[0], raw_image.shape[1],
            #                               scale)

            # Create image with Perlin noise in the red channel
            # blood = np.zeros_like(raw_image)
            # blood[:, :, 2] = np.round(perlin * 255.0).astype(np.uint8)

            # Compute the mean lightness (Value on HSV) of the original tool
            # hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
            # mean = np.mean(hsv[:, :, 2])

            # Add blood reflections to the tool
            # new_image = np.round(.25 * raw_image + .75 * blood).astype(np.uint8)

            # Compute the mean lightness (HSV value) of the new tool
            # new_hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
            # new_mean = np.mean(new_hsv[:, :, 2])

        return new_image, new_label


    @staticmethod
    def contiguous(sides):
        if sides == set(['top', 'right']) \
            or sides == set(['right', 'bottom']) \
            or sides == set(['bottom', 'left']) \
            or sides == set(['left', 'top']):
            return True
        return False


    def tool_zoom(self, factor_range, raw_image, raw_label):
        """
        @brief The size of the tool can change up to 'perc_range' percentage points.

        @param[in]  factor_range  Range of change of the tool size.

        @returns a randomly zoomed image and label.
        """
        # Initially, we just take image and label as they are
        new_image = None
        new_label = None

        # Compute the percentage of change
        min_range = int(round(factor_range[0] * 100.))
        max_range = int(round(factor_range[1] * 100.))
        perc = np.random.randint(min_range, max_range + 1)

        # Perform zoom operation (can be either zoom in or out)
        if perc > 100:
            crop_h = int(round(raw_image.shape[0] / (perc / 100.)))
            crop_w = int(round(raw_image.shape[1] / (perc / 100.)))
            crop_h_half = crop_h // 2
            crop_w_half = crop_w // 2

            # Choose a random point in the tool as crop centre
            tool_pixels = np.nonzero(raw_label)

            #  Remove those points that would make the crop go out of the image
            min_x = crop_w_half
            min_y = crop_h_half
            max_x = raw_image.shape[1] - crop_w_half
            max_y = raw_image.shape[0] - crop_h_half
            coords = [[x, y] for x, y in zip(tool_pixels[1].tolist(), tool_pixels[0].tolist()) \
                if x >= min_x and x <= max_x and y >= min_y and y <= max_y]
            if len(coords):
                centre_idx = np.random.randint(len(coords))
                cx = coords[centre_idx][0]
                cy = coords[centre_idx][1]
            else:
                possible_centres = []
                for x in range(min_x, max_x + crop_w_half, crop_w_half):
                    for y in range(min_y, max_y + crop_h_half, crop_h_half):
                        if np.nonzero(raw_label[y - crop_h_half:y + crop_h_half, x - crop_w_half:x + crop_w_half])[0].shape[0] > 0:
                            possible_centres.append([x, y])
                cx, cy = possible_centres[np.random.randint(len(possible_centres))]

            # Perform the crop
            new_image = raw_image[cy - crop_h_half:cy + crop_h_half,
                                  cx - crop_w_half:cx + crop_w_half]
            new_label = raw_label[cy - crop_h_half:cy + crop_h_half,
                                  cx - crop_w_half:cx + crop_w_half]
        else:
            # Compute the new size
            new_h = int(round(raw_label.shape[0] * np.sqrt(perc / 100.)))
            new_w = int(round(raw_label.shape[1] * np.sqrt(perc / 100.)))

            # Compute the size of the offset (can be a crop or a pad)
            offset_h = np.abs(new_h - raw_label.shape[0])
            offset_w = np.abs(new_w - raw_label.shape[1])

            offset_top, offset_left, offset_bottom, offset_right = \
                CustomBackend.compute_padding_offsets(raw_label, offset_h,
                                                      offset_w)
            new_image, new_label = CustomBackend.pad_top(raw_image, raw_label, offset_top)
            new_image, new_label = CustomBackend.pad_bottom(new_image, new_label, offset_bottom)
            new_image, new_label = CustomBackend.pad_right(new_image, new_label, offset_right)
            new_image, new_label = CustomBackend.pad_left(new_image, new_label, offset_left)

        return new_image, new_label

    @staticmethod
    def compute_cropping_offsets(raw_label, offset_h, offset_w):
        # Compute instrument bounding box
        min_x = np.min(np.where(raw_label)[1])
        max_x = np.max(np.where(raw_label)[1])
        min_y = np.min(np.where(raw_label)[0])
        max_y = np.max(np.where(raw_label)[0])

        # Compute proportions that are free to the sides of the bounding box
        top_prop = float(min_y) / raw_label.shape[0]
        bottom_prop = 1. - top_prop
        left_prop = float(min_x) / raw_label.shape[1]
        right_prop = 1. - left_prop

        # Compute offsets according to proportions
        offset_top = int(round(offset_h * top_prop))
        offset_bottom = offset_h - offset_top
        offset_left = int(round(offset_w * left_prop))
        offset_right = offset_w - offset_left

        # Compute maximum offsets
        max_offset_top = min_y
        max_offset_bottom = raw_label.shape[0] - max_y
        max_offset_left = min_x
        max_offset_right = raw_label.shape[1] - max_x

        # Now the padding depends on which borders the tool is touching
        sides = geometry.entry_sides_in_mask(raw_label)
        if len(sides) == 1: # Only one border touched
            if 'left' in sides:
                offset_left = 0
                offset_right = offset_w
                offset_top = min(offset_top, max_offset_top)
                offset_bottom = min(offset_bottom, max_offset_bottom)
            elif 'top' in sides:
                offset_top = 0
                offset_bottom = offset_h
                offset_left = min(offset_left, max_offset_left)
                offset_right = min(offset_right, max_offset_right)
            elif 'right' in sides:
                offset_right = 0
                offset_left = offset_w
                offset_top = min(offset_top, max_offset_top)
                offset_bottom = min(offset_bottom, max_offset_bottom)
            elif 'bottom' in sides:
                offset_bottom = 0
                offset_top = offset_h
                offset_left = min(offset_left, max_offset_left)
                offset_right = min(offset_right, max_offset_right)
        elif len(sides) == 2 and CustomBackend.contiguous(sides): # Two contiguous borders touched
            if sides == set(['top', 'right']):
                offset_top = 0
                offset_right = 0
                offset_bottom = offset_h
                offset_left = offset_w
            elif sides == set(['right', 'bottom']):
                offset_right = 0
                offset_bottom = 0
                offset_left = offset_w
                offset_top = offset_h
            elif sides == set(['bottom', 'left']):
                offset_bottom = 0
                offset_left = 0
                offset_top = offset_h
                offset_right = offset_w
            elif sides == set(['left', 'top']):
                offset_left = 0
                offset_top = 0
                offset_right = offset_w
                offset_bottom = offset_h
        elif len(sides) == 2 and not CustomBackend.contiguous(sides):
            if sides == set(['top', 'bottom']):
                top_prop = np.random.uniform(0., 1.)
                offset_top = int(round(top_prop * offset_h))
                offset_bottom = offset_h - offset_top
            elif sides == set(['left', 'right']):
                left_prop = np.random.uniform(0., 1.)
                offset_left = int(round(left_prop * offset_w))
                offset_right = offset_w - offset_left
        elif len(sides) == 3:
            if 'top' not in sides:
                offset_top = offset_h
                offset_bottom = 0
                left_prop = np.random.uniform(0., 1.)
                offset_left = int(round(left_prop * offset_w))
                offset_right = offset_w - offset_left
            elif 'left' not in sides:
                offset_left = offset_w
                offset_right = 0
                top_prop = np.random.uniform(0., 1.)
                offset_top = int(round(top_prop * offset_h))
                offset_bottom = offset_h - offset_top
            elif 'bottom' not in sides:
                offset_bottom = offset_h
                offset_top = 0
                left_prop = np.random.uniform(0., 1.)
                offset_left = int(round(left_prop * offset_w))
                offset_right = offset_w - offset_left
            elif 'right' not in sides:
                offset_right = offset_w
                offset_left = 0
                top_prop = np.random.uniform(0., 1.)
                offset_top = int(round(top_prop * offset_h))
                offset_bottom = offset_h - offset_top
        elif len(sides) == 4:
            # Two non-contiguous borders or more than two borders touched
            top_prop = np.random.uniform(0., 1.)
            offset_top = int(round(top_prop * offset_h))
            offset_bottom = offset_h - offset_top
            left_prop = np.random.uniform(0., 1.)
            offset_left = int(round(left_prop * offset_w))
            offset_right = offset_w - offset_left
        else:
            raise ValueError('Unexpected tool mask. Cannot compute \
                             croping offsets.')

        return offset_top, offset_left, offset_bottom, offset_right

    @staticmethod
    def compute_padding_offsets(raw_label, offset_h, offset_w):
        # Compute instrument bounding box
        min_x = np.min(np.where(raw_label)[1])
        min_y = np.min(np.where(raw_label)[0])

        # Compute proportions that are free to the sides of the bounding box
        top_prop = float(min_y) / raw_label.shape[0]
        bottom_prop = 1. - top_prop
        left_prop = float(min_x) / raw_label.shape[1]
        right_prop = 1. - left_prop

        # Compute offsets according to proportions
        offset_top = int(round(offset_h * top_prop))
        offset_bottom = offset_h - offset_top
        offset_left = int(round(offset_w * left_prop))
        offset_right = offset_w - offset_left

        # Now the padding depends on which borders the tool is touching
        sides = geometry.entry_sides_in_mask(raw_label)
        if len(sides) == 1: # Only one border touched
            if 'left' in sides:
                offset_left = 0
                offset_right = offset_w
            elif 'top' in sides:
                offset_top = 0
                offset_bottom = offset_h
            elif 'right' in sides:
                offset_right = 0
                offset_left = offset_w
            elif 'bottom' in sides:
                offset_bottom = 0
                offset_top = offset_h
        elif len(sides) == 2 and CustomBackend.contiguous(sides): # Two contiguous borders touched
            if sides == set(['top', 'right']):
                offset_top = 0
                offset_right = 0
                offset_bottom = offset_h
                offset_left = offset_w
            elif sides == set(['right', 'bottom']):
                offset_right = 0
                offset_bottom = 0
                offset_left = offset_w
                offset_top = offset_h
            elif sides == set(['bottom', 'left']):
                offset_bottom = 0
                offset_left = 0
                offset_top = offset_h
                offset_right = offset_w
            elif sides == set(['left', 'top']):
                offset_left = 0
                offset_top = 0
                offset_right = offset_w
                offset_bottom = offset_h
        else:
            offset_top = 0
            offset_left = 0
            offset_bottom = 0
            offset_right = 0
            common.writeln_warn('More than two contiguous sides touched. Cannot zoom.')
        '''
        elif len(sides) == 2 and not CustomBackend.contiguous(sides):
            if sides == set(['top', 'bottom']):
                offset_top = offset_h / 2
                offset_bottom = offset_top
            elif sides == set(['left', 'right']):
                offset_left = offset_w / 2
                offset_right = offset_left
        elif len(sides) == 3:
            if 'top' not in sides:
                offset_top = offset_h
                offset_bottom = 0
                offset_left = offset_w / 2
                offset_right = offset_w / 2
            elif 'left' not in sides:
                offset_left = offset_w
                offset_right = 0
                offset_top = offset_h / 2
                offset_bottom = offset_h / 2
            elif 'bottom' not in sides:
                offset_bottom = offset_h
                offset_top = 0
                offset_left = offset_w / 2
                offset_right = offset_w / 2
            elif 'right' not in sides:
                offset_right = offset_w
                offset_left = 0
                offset_top = offset_h / 2
                offset_bottom = offset_h / 2
        elif len(sides) == 4:
            # Two non-contiguous borders or more than two borders touched
            offset_top = offset_h / 2
            offset_bottom = offset_h / 2
            offset_left = offset_w / 2
            offset_right = offset_w / 2
        else:
            raise ValueError('Unexpected tool mask. Cannot compute \
                             padding offsets.')
        '''

        return offset_top, offset_left, offset_bottom, offset_right

    @staticmethod
    def pad_top(im, label, pad):
        new_im = np.pad(im, ((pad, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
        new_label = np.pad(label, ((pad, 0), (0, 0)), 'constant', constant_values=(0))
        return new_im, new_label

    @staticmethod
    def pad_bottom(im, label, pad):
        new_im = np.pad(im, ((0, pad), (0, 0), (0, 0)), 'constant', constant_values=(0))
        new_label = np.pad(label, ((0, pad), (0, 0)), 'constant', constant_values=(0))
        return new_im, new_label

    @staticmethod
    def pad_left(im, label, pad):
        new_im = np.pad(im, ((0, 0), (pad, 0), (0, 0)), 'constant', constant_values=(0))
        new_label = np.pad(label, ((0, 0), (pad, 0)), 'constant', constant_values=(0))
        return new_im, new_label

    @staticmethod
    def pad_right(im, label, pad):
        new_im = np.pad(im, ((0, 0), (0, pad), (0, 0)), 'constant', constant_values=(0))
        new_label = np.pad(label, ((0, 0), (0, pad)), 'constant', constant_values=(0))
        return new_im, new_label

    @staticmethod
    def crop_top(im, label, crop):
        if crop != 0:
            return im[crop:,:], label[crop:,:]
        else:
            return im, label

    @staticmethod
    def crop_bottom(im, label, crop):
        if crop != 0:
            return im[:-crop,:], label[:-crop,:]
        else:
            return im, label

    @staticmethod
    def crop_left(im, label, crop):
        if crop != 0:
            return im[:, crop:], label[:, crop:]
        else:
            return im, label

    @staticmethod
    def crop_right(im, label, crop):
        if crop != 0:
            return im[:, :-crop], label[:, :-crop]
        else:
            return im, label

    @staticmethod
    def shift_left(raw_image, raw_label):
        new_image = raw_image
        new_label = raw_label

        # Find centre of mass of the tool
        cm_y, cm_x = scipy.ndimage.center_of_mass(raw_label)
        cm_y = int(round(cm_y))
        cm_x = int(round(cm_x))

        # Compute shift distance
        max_shift = cm_x
        shift = np.random.randint(max_shift)

        # Shift image
        if shift != 0:
            shape_y, shape_x, shape_z = raw_image.shape
            image_x_zeros = np.zeros((shape_y, shift, shape_z), dtype=np.uint8)
            label_x_zeros = np.zeros((shape_y, shift), dtype=np.uint8)
            new_image = np.concatenate((raw_image[:, shift:], image_x_zeros), axis=1)
            new_label = np.concatenate((raw_label[:, shift:], label_x_zeros), axis=1)

        return new_image, new_label

    @staticmethod
    def shift_right(raw_image, raw_label):
        new_image = raw_image
        new_label = raw_label

        # Find centre of mass of the tool
        cm_y, cm_x = scipy.ndimage.center_of_mass(raw_label)
        cm_y = int(round(cm_y))
        cm_x = int(round(cm_x))

        # Compute shift distance
        max_shift = raw_label.shape[1] - cm_x
        shift = np.random.randint(max_shift)

        # Shift image
        if shift != 0:
            shape_y, shape_x, shape_z = raw_image.shape
            image_x_zeros = np.zeros((shape_y, shift, shape_z), dtype=np.uint8)
            label_x_zeros = np.zeros((shape_y, shift), dtype=np.uint8)
            new_image = np.concatenate((image_x_zeros, raw_image[:, :-shift]), axis=1)
            new_label = np.concatenate((label_x_zeros, raw_label[:, :-shift]), axis=1)

        return new_image, new_label

    @staticmethod
    def shift_top(raw_image, raw_label):
        new_image = raw_image
        new_label = raw_label

        # Find centre of mass of the tool
        cm_y, cm_x = scipy.ndimage.center_of_mass(raw_label)
        cm_y = int(round(cm_y))
        cm_x = int(round(cm_x))

        # Compute shift distance
        max_shift = cm_y
        shift = np.random.randint(max_shift)

        # Shift image
        if shift != 0:
            shape_y, shape_x, shape_z = raw_image.shape
            image_y_zeros = np.zeros((shift, shape_x, shape_z), dtype=np.uint8)
            label_y_zeros = np.zeros((shift, shape_x), dtype=np.uint8)
            new_image = np.concatenate((raw_image[shift:], image_y_zeros), axis=0)
            new_label = np.concatenate((raw_label[shift:], label_y_zeros), axis=0)

        return new_image, new_label

    @staticmethod
    def shift_bottom(raw_image, raw_label):
        new_image = raw_image
        new_label = raw_label

        # Find centre of mass of the tool
        cm_y, cm_x = scipy.ndimage.center_of_mass(raw_label)
        cm_y = int(round(cm_y))
        cm_x = int(round(cm_x))

        # Compute shift distance
        max_shift = raw_label.shape[0] - cm_y
        shift = np.random.randint(max_shift)

        # Shift image
        if shift != 0:
            shape_y, shape_x, shape_z = raw_image.shape
            image_y_zeros = np.zeros((shift, shape_x, shape_z), dtype=np.uint8)
            label_y_zeros = np.zeros((shift, shape_x), dtype=np.uint8)
            new_image = np.concatenate((image_y_zeros, raw_image[:-shift]), axis=0)
            new_label = np.concatenate((label_y_zeros, raw_label[:-shift]), axis=0)

        return new_image, new_label

class AugmentationEngine(object):
    BACKENDS = {
        'keras': [                # keras_preprocessing.image.ImageDataGenerator
            'rotation_range',
            'width_shift_range',
            'height_shift_range',
            'brightness_range',
            'shear_range',
            'zoom_range',
            'channel_shift_range',
            'fill_mode',
            'cval',
            'horizontal_flip',
            'vertical_flip',
        ],
        'albumentations': [       # albumentations.augmentations.transforms
            'albu_bg',
            'albu_fg',
            'albu_blend',
        ],
        'custom': [
            'tool_gray',
            'tool_rotation',
            'tool_shift',
            'tool_zoom',
            'blend_border',
            'blood_droplets',
        ],
    }

    def augment(self, dic_of_augmentations, raw_image, raw_label=None):
        """
        @brief Augment the given image and label using the augmentations specified in the dictionary.

        @param[in]  dic_of_augmentations  String -> parameters (usually an integer or a float).
        @param[in]  raw_image             Numpy ndarray BGR image. Shape (height, width, 3).
        @param[in]  raw_label             Numpy ndarray segmentation mask. Shape (height, width).

        @returns (image, label) if a label was given, otherwise just an image.
        """
        assert(len(raw_image.shape) == 3)
        assert(isinstance(raw_image, np.ndarray))
        if raw_label is not None:
            assert(len(raw_label.shape) == 2)
            assert(isinstance(raw_label, np.ndarray))
        new_image = None
        new_label = None

        # Collect custom augmentation options
        custom_dic = {opt : dic_of_augmentations[opt] \
            for opt in dic_of_augmentations if opt in self.BACKENDS['custom']}

        # Collect Keras options
        keras_dic = {opt : dic_of_augmentations[opt] \
            for opt in dic_of_augmentations if opt in self.BACKENDS['keras']}

        # Collect Albumentations options
        albumentations_dic = {opt : dic_of_augmentations[opt] \
            for opt in dic_of_augmentations if opt in self.BACKENDS['albumentations']}

        # Run custom augmentations
        new_image, new_label = AugmentationEngine.custom_backend(custom_dic, raw_image, raw_label)

        # Run Keras augmentations
        new_image, new_label = AugmentationEngine.keras_backend(keras_dic, new_image, new_label)

        # Run Albumentations augmentations
        new_image, new_label = AugmentationEngine.albumentations_backend(albumentations_dic,
            new_image, new_label)

        if raw_label is not None:
            return new_image, new_label
        else:
            return new_image

    @staticmethod
    def keras_backend(keras_dic, raw_image, raw_label=None, seg_thresh=128, proba=None):
        assert(raw_image.shape[0] > 0 and raw_image.shape[1] > 0)
        assert(len(raw_image.shape) == 3)
        # TODO: Implement probability of using augmentation here

        new_image = None
        new_label = None

        # Convert image to RGB and expand it, the Keras preprocessing module expects so
        x = np.expand_dims(raw_image, 0)[..., ::-1]
        if raw_label is not None:
            y = np.expand_dims(cv2.cvtColor(raw_label, cv2.COLOR_GRAY2BGR), 0)

        # Create Keras preprocessing ImageDataGenerator
        datagen = KerasGenerator(**keras_dic)

        # Perform same augmentation on both image and label
        seed = np.random.randint(2**32)
        it = datagen.flow(x, batch_size=1, seed=seed)
        new_image = it.next()[0].astype('uint8')[..., ::-1] # Convert image back to BGR

        if raw_label is not None:
            it = datagen.flow(y, batch_size=1, seed=seed)
            new_label = it.next()[0].astype('uint8')

            # Threshold the mask, augmentation can introduce new pixel intensities
            new_label[new_label < seg_thresh] = 0
            new_label[new_label >= seg_thresh] = 255
            new_label = cv2.cvtColor(new_label, cv2.COLOR_BGR2GRAY)

        return new_image, new_label

    @staticmethod
    def albumentations_backend(albumentations_dic, raw_image, raw_label=None, proba=None):
        new_image = raw_image
        new_label = raw_label
        augmentation = None

        if albumentations_dic:
            # Background augmentation configuration
            if 'albu_bg' in albumentations_dic:
                albu_bg = albumentations.Compose([
                    albumentations.RandomRotate90(p=1.0),
                ], p=albumentations_dic['albu_bg'])
                augmentation = albu_bg

            # Foreground augmentation configuration
            elif 'albu_fg' in albumentations_dic:
                albu_fg = albumentations.Compose([
                    # albumentations.OneOf([
                        # albumentations.ToGray(p=1.0),
                        # albumentations.ChannelShuffle(p=1.0),
                        # albumentations.RandomRain(slant_lower=-10, slant_upper=10, drop_length=5,
                        #    drop_width=5, drop_color=(0, 0, 255), blur_value=7,
                        #    brightness_coefficient=0.7, rain_type='drizzle', p=1.0)
                    #], p=1.0),
                ], p=albumentations_dic['albu_fg'])
                augmentation = albu_fg

            # Blending augmentation configuration
            elif 'albu_blend' in albumentations_dic:
                albu_blend = albumentations.Compose([

                    # Illumination, brightness, contrast
                    # albumentations.OneOf([
                    #    albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    #], p=0.5),

                    # Artifacts
                    albumentations.OneOf([
                        albumentations.Cutout(num_holes=np.random.randint(8, 64),
                                              max_h_size=np.random.randint(8, 32),
                                              max_w_size=np.random.randint(8, 32),
                                              fill_value=[np.random.randint(0, 256),
                                                          np.random.randint(0, 256),
                                                          np.random.randint(0, 256)],
                                              p=1.0),
                        albumentations.RandomShadow(p=1.0),
                        albumentations.RandomFog(p=1.0),
                        albumentations.JpegCompression(quality_lower=20, quality_upper=100, p=1.0),
                        # albumentations.RandomSnow(p=1.0),
                    ], p=0.5),

                    # Colour
                    albumentations.OneOf([
                        albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20,
                                                b_shift_limit=20, p=1.0),
                        albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                                          val_shift_limit=20, p=1.0),
                        # albumentations.ChannelDropout(channel_drop_range=(1, 1),
                        #                              fill_value=np.random.randint(0, 128), p=1.0),
                        # albumentations.ChannelShuffle(p=1.0),
                    ], p=0.5),

                    # Noise or blur
                    albumentations.OneOf([
                        albumentations.OneOf([
                            albumentations.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True,
                                                               p=1.0),
                            albumentations.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True,
                                                               p=1.0),
                            albumentations.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),
                            albumentations.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5),
                                                    p=1.0),
                        ], p=1.0),
                        albumentations.OneOf([
                            albumentations.Blur(blur_limit=7, p=1.0),
                            albumentations.GaussianBlur(blur_limit=7, p=1.0),
                            albumentations.MotionBlur(blur_limit=7, p=1.0),
                            albumentations.MedianBlur(blur_limit=7, p=1.0),
                            # albumentations.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode='fast',
                            #                         p=1.0),
                        ], p=1.0),
                    ], p=0.5),

                ], p=albumentations_dic['albu_blend'])
                augmentation = albu_blend
            else:
                common.writeln_warn('Unknown albumentation augmentation option.')

            # Albumentations requires you to provide an input dictionary, let's make it
            if isinstance(raw_image, dict):
                assert('image' not in raw_image)
                data = { 'mask': raw_label }

                # Perform data augmentation for all blending modes
                new_image = {}
                for k in raw_image:
                    data['image'] = raw_image[k]
                    random_st0 = random.getstate()
                    np_random_st0 = np.random.get_state()
                    augmented = augmentation(**data)
                    np.random.set_state(np_random_st0)
                    random.setstate(random_st0)
                    new_image[k] = augmented['image']
            else:
                data = { 'image': raw_image }
                if raw_label is not None:
                    data['mask'] = raw_label
                augmented = augmentation(**data)
                new_image = augmented['image']

            # As the transformation should be the same for all blending modes 
            # we can get the last mask for all the images
            new_label = augmented['mask'] if raw_label is not None else None

        return new_image, new_label

    @staticmethod
    def custom_backend(custom_dic, raw_image, raw_label=None, proba=None):
        # TODO: Implement probability of using augmentation here

        assert(isinstance(raw_image, np.ndarray))
        if raw_label is not None:
            assert(isinstance(raw_label, np.ndarray))

        cb = CustomBackend(custom_dic)
        new_image, new_label = cb.augment(raw_image, raw_label)

        return new_image, new_label

class Foreground:
    # FG_ACCEPTED_AUG = ['horizontal_flip', 'vertical_flip']

    """
    @class Foreground holds a foreground image and its segmentation mask.
    """

    def __init__(self, frame, seg):
        assert(frame is not None)
        assert(seg is not None)
        self.frame = frame
        self.seg = seg

    def resize_to_width(self, width, interp):
        self.frame.resize_to_width(width, interp)
        self.seg.resize_to_width(width, cv2.INTER_NEAREST)

    def random_crop(self, new_h, new_w, attempts=5):

        # The random crop has to take foreground pixels, it cannot leave
        # just background
        successful_crop = False

        while not successful_crop and attempts > 0:
            attempts -= 1
            # Perform a random crop
            im, mask = \
                image.CaffeinatedAbstract.random_crop_same_coord_list(
                    [self.frame.raw, self.seg.raw], new_h, new_w)

            if np.nonzero(mask)[0].shape[0] > 0:
                successful_crop = True

        if successful_crop:
            self.frame._raw_frame = im
            self.seg._raw_frame = mask
        else:
            common.writeln_warn('The foreground crop did not keep tool pixels.')

        return successful_crop

    def augment(self, dic_of_augmentations):
        assert(self.frame.raw is not None)
        assert(self.seg.raw is not None)

        # Augment image with the methods specified by the user
        aug_eng = AugmentationEngine()
        new_im, new_seg = aug_eng.augment(dic_of_augmentations, self.frame.raw, self.seg.raw)

        # Update internal tool image and segmentation mask
        self.frame = image.CaffeinatedImage(new_im, self.frame.name)
        self.seg = image.CaffeinatedLabel(new_seg, self.seg.name, self.seg.classes,
            self.seg.class_map)
    
    @property
    def shape(self):
        return self.frame.shape

"""
@class Background is just pretty much CaffeinatedImage with Keras preprocessing augmentation.
"""
class Background(image.CaffeinatedImage):
    # BG_ACCEPTED_AUG = ['horizontal_flip', 'vertical_flip']

    def __init__(self, raw_frame, name, label=None):
        super(Background, self).__init__(raw_frame, name, label)


    def augment(self, dic_of_augmentations):
        # TODO: CustomBackend is not supported here yet

        # Collect Keras options
        keras_backend_dic = { k : dic_of_augmentations[k] \
            for k in dic_of_augmentations if k in AugmentationEngine.BACKENDS['keras'] }

        # Collect Albumentations options
        albumentations_dic = {k : dic_of_augmentations[k] \
            for k in dic_of_augmentations if k in AugmentationEngine.BACKENDS['albumentations']}

        # Perform Keras augmentations
        self._raw_frame, _ = AugmentationEngine.keras_backend(keras_backend_dic, self.raw)

        # Perform Albumentation augmentations
        self._raw_frame, _ = AugmentationEngine.albumentations_backend(albumentations_dic,
                                                                       self.raw)


class BlendedImageSet:
    """
    @class BlendedImageSet represents a set of blended images, given a fg/bg pair.
    """

    def __init__(self, bg, fgs):
        self.bg = bg
        self.fgs = fgs
        self.blended_images = {}
        self.flying_distractors = [] # Holds pairs of [image, mask]

    def add_flying_distractor(self, bg, fg, aug_dic):
        """
        @brief Add an artifact to the image with the shape of a tool and the texture of a
               background.
        @param[in]  bg  Background object.
        @param[in]  fg  Foreground object.
        @returns nothing.
        """
        fg_im = fg.frame.raw
        fg_mask = fg.seg.raw

        # Get dictionary of custom augmentations applied to the foreground
        custom_backend_dic = { k: aug_dic[k] for k in aug_dic if k in
            AugmentationEngine.BACKENDS['custom'] }

        # Remove those augmentations that are ok for tools but not suitable for distractors
        custom_backend_dic.pop('tool_gray', None)

        # Apply custom backend augmentations
        cb = CustomBackend(custom_backend_dic)
        fg_im, fg_mask = cb.augment(fg_im, fg_mask)

        # Collect Keras options and perform augmentations on the flying distractor
        keras_backend_dic = { k : aug_dic[k] for k in aug_dic if k in
            AugmentationEngine.BACKENDS['keras'] }
        fg_im, fg_mask = AugmentationEngine.keras_backend(keras_backend_dic, fg_im, fg_mask)

        # FIXME: Apply albumentations that are applied to foreground tools

        # Resize the flying image and mask to the standard size
        h = self.bg.raw.shape[0]
        w = self.bg.raw.shape[1]
        flying_image = cv2.resize(bg.raw, (w, h))
        flying_mask = cv2.resize(fg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Add pair to the list of flying distractors
        self.flying_distractors.append([flying_image, flying_mask])

    def blend(self, blend_modes):
        cy = int(round(.5 * self.bg._raw_frame.shape[0]))
        cx = int(round(.5 * self.bg._raw_frame.shape[1]))
        for mode in blend_modes:
            blend = self.bg.raw.copy()
            label = np.zeros((self.bg.shape[0], self.bg.shape[1]), dtype=np.uint8)

            # Blend flying distractors on the background
            for [fl_im, fl_mask] in self.flying_distractors:
                blend = blending.blend(fl_im, fl_mask, blend.copy(), cy, cx, mode)

            # Blend instruments
            for fg in self.fgs:
                blend = blending.blend(fg.frame.raw.copy(), fg.seg.raw.copy(), blend.copy(), cy, cx, mode)
                label = cv2.bitwise_or(label, fg.seg.raw)

            self.blended_images[mode] = [blend, label]
        # return self.blended_images

    def augment(self, dic_of_aug,
            same_for_all_modes=['blend_border', 'albu_blend']):
        # Prepare the dictionary with those augmentations that must be the same for all
        # blending modes
        same_aug = {k: dic_of_aug[k] for k in dic_of_aug if k in same_for_all_modes}

        # Set random state that will be used to enforce the same augmentations across modes
        rs = np.random.RandomState(None)

        # Prepare backends to perform the same augmentations on all modes
        cb = CustomBackend(same_aug, rs)

        # Apply custom augmentations, identical for all blending modes
        for mode in self.blended_images:
            new_image = self.blended_images[mode][0]
            new_label = self.blended_images[mode][1]

            # Apply augmentation freezing the random state so that the
            # augmentations of each mode are the same
            st0 = cb.rs.get_state()
            new_image, new_label = cb.augment(new_image, new_label)
            cb.rs.set_state(st0)

            # Save blended images
            self.blended_images[mode] = [new_image, new_label]

        # Apply albumentations, identical for all blending modes
        blended_images_for_albu = { k: self.blended_images[k][0] for k in self.blended_images }
        new_blended_images, new_label = AugmentationEngine.albumentations_backend(same_aug,
            blended_images_for_albu, new_label)
        self.blended_images = { k: [new_blended_images[k], new_label] for k in new_blended_images }

        # TODO: Apply augmentations that can be different in each blending mode

        '''
        # Apply augmentation
        rs=np.random.RandomState(None)
        blend_border_param = None
        if 'blend_border' in dic_of_augmentations:
            for mode in self.blended_images:
                im = self.blended_images[mode][0]
                label = self.blended_images[mode][1]
                st0 = rs.get_state()
                new_image, new_label = cb.blend_border(dic_of_augmentations['blend_border'], im, label, rs=rs)
                rs.set_state(st0)
                self.blended_images[mode] = [new_image, new_label]
            blend_border_param = dic_of_augmentations.pop('blend_border')

        # Custom backend augmentation is applied to each of the blended images
        new_blended_images = {}
        for mode in self.blended_images:
            im = self.blended_images[mode][0]
            label = self.blended_images[mode][1]
            new_image, new_label = cb.augment(im, label)
            new_blended_images[mode] = [new_image, new_label]
        self.blended_images = new_blended_images

        if blend_border_param is not None:
            dic_of_augmentations['blend_border'] = blend_border_param
        '''

class ImageGenerator:
    """
    @class ImageGenerator generates images of background tissue with tools blended on top.
    """
    def __init__(self, bg_list, fg_list, gt_suffix='_seg', gt_ext='.png', classes=2,
            class_map={0: 0, 255: 1}, min_ninst=1, max_ninst=3, fg_aug={}, bg_aug={}, blend_aug={}):
        """
        @param[in]  bg_list    List of paths to background images.
        @param[in]  fg_list    List of paths to foreground images.
        @param[in]  gt_suffix  Suffix of the foreground segmentation images.
        @param[in]  gt_ext     Extension of the segmentation files, typically '.png'.
        @param[in]  classes    Number of classes in the segmentation labels, default 2.
        @param[in]  fg_aug     Dictionary of augmentations for the foreground.
        @param[in]  bg_aug     Dictionary of augmentations for the background.
        @param[in]  blend_aug  Dictionary of augmentations for the blended images.
        """
        self.bg_list = bg_list
        self.fg_list = fg_list
        self.gt_suffix = gt_suffix
        self.gt_ext = gt_ext
        self.classes = classes
        self.class_map = class_map
        self.min_ninst = min_ninst
        self.max_ninst = max_ninst
        self.fg_aug = fg_aug
        self.bg_aug = bg_aug
        self.blend_aug = blend_aug

    def load_fg(self, attempts=5):
        # There are some images in the dataset that do not have tools, this might be because the
        # images actually do not have tools, or because the HSV gilter + GrabCut failed
        failed = True
        while failed and attempts > 0:
            # Choose a random file from the list of foregrounds
            fg_path = np.random.choice(self.fg_list)
            fg_name = common.get_fname_no_ext(fg_path)
            fg_ext = common.get_ext(fg_path)
            fg_gt_path = os.path.join(os.path.dirname(fg_path), fg_name + self.gt_suffix + self.gt_ext)

            # Read image and ground truth segmentation
            frame = image.CaffeinatedImage.from_file(fg_path, fg_name + fg_ext)
            seg = image.CaffeinatedLabel.from_file(fg_gt_path, fg_name + self.gt_suffix + fg_ext,
                    self.classes, self.class_map)

            # Just in case the foreground image is BGRA, we get just the BGR part
            frame._raw_frame = frame._raw_frame[:,:, :3]

            # Check that the image has at least a tool, otherwise pick another foreground
            if np.count_nonzero(seg.raw) > 0:
                failed = False
            else:
                attempts -= 1

        return Foreground(frame, seg)

    def load_bg(self):
        bg_path = np.random.choice(self.bg_list)
        bg_name = common.get_fname_no_ext(bg_path)
        bg_ext = common.get_ext(bg_path)
        im = Background.from_file(bg_path, bg_name + bg_ext)
        im._raw_frame = cv2.cvtColor(im._raw_frame, cv2.COLOR_BGRA2BGR)
        return im


    def get_list_of_foregrounds(self, ninst, min_pixels=2**10):
        # Get a list of foregrounds
        fgs = []
        inst_added = 0
        while inst_added < ninst:
            # Load foreground image
            raw_fg = self.load_fg()

            # Convert mask to 0/1
            label = np.zeros_like(raw_fg.seg.raw, dtype=np.uint8)
            label[raw_fg.seg.raw != 0] = 1

            # Detect connected components
            _, cc = cv2.connectedComponents(label)
            cc_list = []
            for c in range(1, np.max(cc) + 1):
                if np.count_nonzero(cc == c) > min_pixels:
                    new_cc = np.zeros_like(cc, dtype=np.uint8)
                    new_cc[cc == c] = 255
                    cc_list.append(new_cc)

            # Add as many connected components (foreground tools) as requested by the user
            while inst_added < ninst and len(cc_list) > 0:
                inst_added += 1
                frame = raw_fg.frame
                seg = image.CaffeinatedLabel(cc_list.pop(), raw_fg.seg.name, raw_fg.seg.classes,
                    raw_fg.seg.class_map)
                fgs.append(Foreground(frame, seg))
        return fgs


    def generate_image(self, blend_modes, height=None, width=None, margin=1, dilate_ksize=3):
        # Number of instruments is random
        ninst = np.random.randint(self.min_ninst, self.max_ninst + 1)

        # Get a random background
        bg = self.load_bg()

        # If any augmentation specified for the background, we pass it
        if self.bg_aug:
            bg.augment(self.bg_aug)

        # Get a list of foregrounds
        fgs = self.get_list_of_foregrounds(ninst)

        # If any augmentation is specified for the foreground, we pass it
        if self.fg_aug:
            for fg in fgs:
                fg.augment(self.fg_aug)

        # Resize background and foregrounds to the standard width
        bg.resize_to_width(width, cv2.INTER_LANCZOS4)
        for fg in fgs:
            fg.resize_to_width(width, cv2.INTER_LANCZOS4)

        # Random crop foregrounds and background to the minimum size between fg and bg
        fg_min_height = min([fg.shape[0] for fg in fgs])
        fg_min_width = min([fg.shape[1] for fg in fgs])
        min_height = min(fg_min_height, bg.shape[0])
        min_width = min(fg_min_width, bg.shape[1])
        new_fgs = []
        for fg in fgs:
            # It is possible that the random crop does not pick tool, then we discard
            # this foreground
            if fg.random_crop(min_height, min_width):
                new_fgs.append(fg)
            else:
                common.writeln_warn('Foreground discarded as random crop left an empty image.')
        fgs = new_fgs
        bg.random_crop(min_height, min_width)

        # It is possible that the random cropping did not catch a foreground, so we amend it
        # by adding new instruments until we reach the amount we want
        while len(fgs) < ninst:
            extra_fg = self.get_list_of_foregrounds(1)[0]
            extra_fg.augment(self.fg_aug)
            extra_fg.resize_to_width(width, cv2.INTER_LANCZOS4)
            if extra_fg.frame.raw.shape[0] >= min_height \
                    and extra_fg.random_crop(min_height, min_width):
                fgs.append(extra_fg)

        # Blend images with the requested modes
        blend_set = BlendedImageSet(bg, fgs)
        blend_set.add_flying_distractor(self.get_flying_distractor_texture(), self.load_fg(),
                                        self.fg_aug)
        blend_set.blend(blend_modes)
        blend_set.augment(self.blend_aug)
        blended_images = blend_set.blended_images

        return blended_images

    def get_flying_distractor_texture(self, p=0.8, tol=10):
        '''
        @brief Gets an image that will be later on cut out with the shape of a tool. It chooses
               randomly among two options: a background image or the green part of a green screen.
        @returns an image as a numpy ndarray, shape (h, w, 3).
        '''
        texture_im = None
        if np.random.binomial(1, p):
            texture = self.load_bg()
        else:
            # Load a foreground
            fg_path = np.random.choice(self.fg_list)
            fg_name = common.get_fname_no_ext(fg_path)
            fg_ext = common.get_ext(fg_path)
            fg_gt_path = os.path.join(os.path.dirname(fg_path),
                                      fg_name + self.gt_suffix + self.gt_ext)
            im = cv2.imread(fg_path)
            mask = cv2.imread(fg_gt_path, 0)
            min_x = np.min(np.where(mask)[1]) - tol
            max_x = np.max(np.where(mask)[1]) + tol
            min_y = np.min(np.where(mask)[0]) - tol
            max_y = np.max(np.where(mask)[0]) + tol
            min_x = max(0, min_x)
            max_x = min(im.shape[1] - 1, max_x)
            min_y = max(0, min_y)
            max_y = min(im.shape[0] - 1, max_y)
            im[min_y:max_y + 1, min_x:max_x + 1] = [0, 255, 0]
            mask[min_y:max_y + 1, min_x:max_x + 1] = 255

            # Inpaint tool with information from green screen
            texture_im = cv2.inpaint(im, mask, 3, cv2.INPAINT_TELEA)
            texture = Background(texture_im, 'texture_im')

        return texture

if __name__ == "__main__":
    raise RuntimeError('image_generator is a module, not a script.')
