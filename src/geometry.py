"""
@brief   Geometry module.
@details This module contains functions that are helpful for geometric calculations.
@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    15 Apr 2017.
"""

import numpy as np
import cv2


#
# @brief Compute the number of entry points in a binary mask.
#
# @param[in] mask   Binary mask to be evaluated.
# @param[in] margin Size of the border to take into account for connected components.
#
# @returns a mask with background labelled as zero, and each connected component with a different
#          number (starting in one).
def entry_points_in_mask(mask, margin, dilate_ksize=0):
    # Dilate if the user wants, just to avoid pixel errors in the mask to generate two segments
    # because the entrypoint is mistakenly disconnected
    if dilate_ksize > 2:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        new_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        new_mask = mask

    # Assert that the mask is binary
    uniq = np.unique(new_mask)
    assert(uniq.shape[0] == 1 or uniq.shape[0] == 2)
    if uniq.shape[0] == 2:
        assert((uniq[0] == 0 and uniq[1] == 1)
               or (uniq[0] == 1 and uniq[1] == 0))

    # Remove the central part of the mask
    border_mask = new_mask.copy()
    border_mask[margin:-margin, margin:-margin] = 0

    # Compute the connected components
    _, markers = cv2.connectedComponents(border_mask)

    return markers


#
# @brief Get a list of masks, each with a continuous entrypoint segment.
#
# @param[in] mask   Binary mask to be evaluated.
# @param[in] margin Size of the border to take into account for connected components.
#
# @returns an integer that represents the number of entry points.
def no_entry_points_in_mask(mask, margin):
    # Assert that the mask is binary
    uniq = np.unique(mask)
    assert(uniq.shape[0] == 1 or uniq.shape[0] == 2)
    if uniq.shape[0] == 2:
        assert((uniq[0] == 0 and uniq[1] == 1)
               or (uniq[0] == 1 and uniq[1] == 0))

    # Remove the central part of the mask
    border_mask = mask.copy()
    border_mask[margin:-margin, margin:-margin] = 0

    # Compute the connected components
    _, markers = cv2.connectedComponents(border_mask)

    return np.amax(markers)


#
# @brief Finds out which boundaries are being touched by the mask.
#
# @param[in] mask         Binary mask to be analysed.
# @param[in] dilate_ksize If <3, no dilation is used. If >= 3, a dilation kernel is used.
#
# @returns a list of strings that could contain all or just some of these: 'top', 'bottom',
#          'left', 'right'.
def entry_sides_in_mask(mask, dilate_ksize=0):
    if dilate_ksize > 2:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        new_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        new_mask = mask

    sides = set()

    # Does it touch top?
    if np.nonzero(new_mask[0, :])[0].shape[0] > 0:
        sides.add('top')

    # Does it touch left?
    if np.nonzero(new_mask[:, 0])[0].shape[0] > 0:
        sides.add('left')

    # Does it touch right?
    if np.nonzero(new_mask[:, -1])[0].shape[0] > 0:
        sides.add('right')

    # Does it touch bottom?
    if np.nonzero(new_mask[-1, :])[0].shape[0] > 0:
        sides.add('bottom')

    return sides


#
# @brief Detects if a contour is in contact with the border or not.
#
# @param[in] cnt OpenCV contour.
# @param[in] img Input frame.
#
# @returns True if the given contour is in contact with the image border.
#          Otherwise returns false.
def contour_touches_border(cnt, img):

    x, y, w, h = cv2.boundingRect(cnt)
    xmin = 0
    ymin = 0
    xmax = img.shape[1] - 1
    ymax = img.shape[0] - 1

    return x <= xmin or y <= ymin or w >= xmax or h >= ymax


#
# @brief Compute the number of connected components in the image.
#
# @param[in] mask Binary image.
#
# @returns an integer with the number of CC.
def no_cc(mask):
    assert(mask.dtype == np.uint8)
    _, markers = cv2.connectedComponents(mask)
    return np.amax(markers)


#
# @brief Returns a mask with the largest connected components.
#
# @param[in] mask Input binary mask with noise.
# @param[in] ncc  Number of connected components to be retrieved.
#
# @returns a clean mask with just the connected components of largest area.
def largest_cc_mask(mask, ncc=1):
    assert(mask.dtype == np.uint8)

    # Label connected components
    _, markers = cv2.connectedComponents(mask)
    frequency = np.bincount(markers.flatten()).tolist()

    # Get largest connected components
    largest_cc = [x for _, x in sorted(
                  zip(frequency, range(len(frequency))))][::-1][1:][:ncc]

    # Create the new mask
    new_mask = np.zeros_like(mask)
    for i in largest_cc:
        new_mask[markers == i] = 255

    return new_mask


if __name__ == '__main__':
    raise RuntimeError(
        'Error, this module is not supposed to be run as a script.')
