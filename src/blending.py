"""
@brief   Module that stores many blending functions.
@author  Luis Carlos Garcia-Peraza Herrera
         (luiscarlos.gph@gmail.com)
@date    7 Mar 2019.
"""

import numpy as np
import cv2
import blend_modes


# 
# @brief Converts the given parameters into a foreground and a background as expected by the
#        blend modes module.
#
def bm_preprocess(obj, mask, dst, cy, cx):
    
    # Erode mask to remove green pixels at the border
    kernel = np.ones((3, 3), np.uint8)
    emask = cv2.erode(mask, kernel, iterations = 1)
    
    # Convert eroded mask into a 3-channel floating image
    emask = emask.astype(np.float64) / 255
    emask = emask[:,:, None] * np.ones(4, dtype = np.float64)[None, None,:]

    # Convert everything to float
    fg = cv2.cvtColor(obj, cv2.COLOR_BGR2RGBA).astype(np.float64)
    bg = cv2.cvtColor(dst, cv2.COLOR_BGR2RGBA).astype(np.float64) 

    # Multiply fg with the alpha matte
    fg = cv2.multiply(emask, fg)
    
    # Compute the embedding coordinates
    t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = embed_coord(fg, bg, cy, cx)
    
    return fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg

# 
# @brief   Hard light blending. 
# @details Multiplies or screens the colors, depending on the blend color. The effect is similar 
#          to shining a harsh spotlight on the image. If the blend color (light source) is lighter 
#          than 50% gray, the image is lightened, as if it were screened. This is useful for adding 
#          highlights to an image. If the blend color is darker than 50% gray, the image is 
#          darkened, as if it were multiplied. This is useful for adding shadows to an image. 
#          Painting with pure black or white results in pure black or white.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj. Must have the same size as obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def hard_light_blending(obj, mask, dst, cy, cx, opacity = 1.0):

    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    

    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.hard_light(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)

    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def lighthen_only_blending(obj, mask, dst, cy, cx, opacity = 1.0):

    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    
    
    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.lighten_only(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def soft_light_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    

    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.soft_light(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def dodge_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    
    
    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.dodge(bg[t_bg:b_bg, l_bg:r_bg,:], 
        fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def addition_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    

    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.addition(bg[t_bg:b_bg, l_bg:r_bg,:], 
        fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def darken_only_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    
    
    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.darken_only(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def multiply_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    

    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.multiply(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def difference_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    
    
    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.difference(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def subtract_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    
    
    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.subtract(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def grain_extract_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    
    
    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.grain_extract(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def grain_merge_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    
    
    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.grain_merge(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief Blends obj onto dst.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def divide_blending(obj, mask, dst, cy, cx, opacity = 1.0):
    
    fg, bg, t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = bm_preprocess(obj, mask, dst, cy, cx)    

    bg[t_bg:b_bg, l_bg:r_bg] = blend_modes.divide(bg[t_bg:b_bg, 
        l_bg:r_bg,:], fg[t_fg:b_fg, l_fg:r_fg], opacity)
    bg = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    
    return bg

# 
# @brief   Computes the right coordinates to overlay one image on another.
# @details When you want to overlay an image fg over bg at the point (cx, cy) of bg, there
#          might be the need to cut the fg, as it goes out of the bg. This functions computes the
#          exact positions of the two rectangles, in bg and fg, so that you can use this
#          coordinates to replace it like this: 
#          
#          bg[t_bg:b_bg, l_bg:r_bg] = fg[t_fg:b_fg, l_fg:r_fg]
#
# @returns a tuple (t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg) with the top, left, bottom,
#          right coordinates of background and foreground.
def embed_coord(fg, bg, cy, cx):
    
    # Calculate offset of the fg coordinate frame
    off_y = cy - int(round(fg.shape[0] / 2.))
    off_x = cx - int(round(fg.shape[1] / 2.))

    # Compute coordinates for the background
    t_bg = max(off_y, 0)
    l_bg = max(off_x, 0) 
    b_bg = min(off_y + fg.shape[0], bg.shape[0])
    r_bg = min(off_x + fg.shape[1], bg.shape[1])
    
    # Compute the coordinates for the foreground
    t_fg = max(t_bg - off_y, 0)
    l_fg = max(l_bg - off_x, 0)
    b_fg = min(b_bg - off_y, fg.shape[0])
    r_fg = min(r_bg - off_x, fg.shape[1])
    
    # Sanity check: the fg chunk must have the same size as the bg that is going to be replaced
    assert(b_bg - t_bg == b_fg - t_fg)
    assert(r_bg - l_bg == r_fg - l_fg)
    
    return (t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg)

# 
# @brief Blends obj onto dst using a weighted average defined by the mask provided.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def alpha_blending(obj, mask, dst, cy, cx):
    assert(obj.shape[0] > 0 and obj.shape[0] == mask.shape[0])
    assert(obj.shape[1] > 0 and obj.shape[1] == mask.shape[1])
    assert(dst.shape[0] > 0)
    assert(dst.shape[1] > 0)
    
    # Erode mask to remove green pixels at the border
    kernel = np.ones((3, 3), np.uint8)
    emask = cv2.erode(mask, kernel, iterations = 1)
    
    # Convert eroded mask into a 3-channel floating image
    emask = emask.astype(np.float64) / 255
    emask = emask[:,:, None] * np.ones(3, dtype = np.float64)[None, None,:]

    # Convert everything to float
    fg = obj.astype(np.float64)
    bg = dst.astype(np.float64)
    
    # Crop the foreground in case it goes out of the background
    t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = embed_coord(fg, bg, cy, cx)
    
    # Multiply fg and bg with the alpha matte
    fg = cv2.multiply(emask, fg)
    bg[t_bg:b_bg, l_bg:r_bg,:] = cv2.multiply(1.0 - emask[t_fg:b_fg, l_fg:r_fg,:], 
        bg[t_bg:b_bg, l_bg:r_bg,:])
    
    # Combine images in an additive manner
    bg[t_bg:b_bg, l_bg:r_bg,:] = cv2.add(fg[t_fg:b_fg, l_fg:r_fg,:], bg[t_bg:b_bg, l_bg:r_bg,:])
    bg = bg.astype(np.uint8)

    return bg

# 
# @brief Blends obj onto dst using a weighted average defined by the mask provided.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def gaussian_blending(obj, mask, dst, cy, cx, erode_ksize = 3, blur_ksize = 5):
    
    # Erode mask to remove green pixels at the border
    kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
    emask = cv2.erode(mask, kernel, iterations = 1)
    
    # Blur the mask
    emask = cv2.GaussianBlur(emask, (blur_ksize, blur_ksize), 0)

    # Convert eroded mask into a 3-channel floating image
    emask = emask.astype(np.float64) / 255
    emask = emask[:,:, None] * np.ones(3, dtype = np.float64)[None, None,:]

    # Convert everything to float
    fg = obj.astype(np.float64)
    bg = dst.astype(np.float64) 
    
    # Crop the foreground in case it goes out of the background
    t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = embed_coord(fg, bg, cy, cx)
    
    # Multiply fg and bg with the alpha matte
    fg = cv2.multiply(emask, fg)
    bg[t_bg:b_bg, l_bg:r_bg,:] = cv2.multiply(1.0 - emask[t_fg:b_fg, l_fg:r_fg,:], 
        bg[t_bg:b_bg, l_bg:r_bg,:])
    
    # Combine images in an additive manner
    bg[t_bg:b_bg, l_bg:r_bg,:] = cv2.add(fg[t_fg:b_fg, l_fg:r_fg,:], bg[t_bg:b_bg, l_bg:r_bg,:])
    bg = bg.astype(np.uint8)

    return bg


# 
# @brief Blends obj onto dst using a weighted average defined by the mask provided.
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def blood_blending(obj, mask, dst, cy, cx, erode_ksize = 3, blur_ksize = 11):
    
    # Erode mask to remove green pixels at the border
    kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
    emask = cv2.erode(mask, kernel, iterations = 1)
    
    # Blur the mask
    emask = cv2.GaussianBlur(emask, (blur_ksize, blur_ksize), 0)

    # Convert eroded mask into a 3-channel floating image
    emask = emask.astype(np.float64) / 255
    emask = emask[:,:, None] * np.ones(3, dtype = np.float64)[None, None,:]

    # Convert everything to float
    fg = obj.astype(np.float64)
    bg = dst.astype(np.float64) 
    
    # Crop the foreground in case it goes out of the background
    t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = embed_coord(fg, bg, cy, cx)
    
    # Multiply fg and bg with the alpha matte
    fg = cv2.multiply(emask, fg)
    bg[t_bg:b_bg, l_bg:r_bg,:] = cv2.multiply(1.0 - emask[t_fg:b_fg, l_fg:r_fg,:], 
        bg[t_bg:b_bg, l_bg:r_bg,:])
    
    # Combine images in an additive manner
    bg[t_bg:b_bg, l_bg:r_bg,:] = cv2.add(fg[t_fg:b_fg, l_fg:r_fg,:], bg[t_bg:b_bg, l_bg:r_bg,:])
    bg = bg.astype(np.uint8)

    return bg


def poisson_normal_blending(obj, mask, dst, cy, cx):
    return poisson_blending(obj, mask, dst, cy, cx, cv2.NORMAL_CLONE)

def poisson_mixed_blending(obj, mask, dst, cy, cx):
    return poisson_blending(obj, mask, dst, cy, cx, cv2.MIXED_CLONE)

# 
# @brief   Blends obj onto dst using Poisson Image Editing.
# @details OpenCV changes the mask to a bounding box around it, be careful when using 
#          seamlessClone().
# 
# @param[in] obj  Foreground image.
# @param[in] mask Binary mask for obj.
# @param[in] dst  Background image.
# @param[in] cy   Row that indicates the point in dst where the centre of obj will be located.
# @param[in] cx   Column that indicates the point in dst where the centre of obj will be located.
#
# @returns an image (np.uint8).
def poisson_blending(obj, mask, dst, cy, cx, clone_type = cv2.NORMAL_CLONE):
    
    '''
    # Compute the obj and mask that actually fit inside dst given the (cx, cy) coordinates
    t_bg, l_bg, b_bg, r_bg, t_fg, l_fg, b_fg, r_fg = embed_coord(obj, dst, cy, cx)
    inner_obj = obj[t_fg + 1:b_fg - 1, l_fg + 1:r_fg - 1]
    inner_mask = mask[t_fg + 1:b_fg - 1, l_fg + 1:r_fg - 1]

    # Now, from the obj that fits in dst, get the part that actually has foreground,
    # seamlessClone() crashes as they modify the ROI to the rect that contains the white part 
    # in the given mask, hence, we have to do the same here before it crashes
    tlx, tly, rw, rh = cv2.boundingRect(inner_mask)
    actual_obj = inner_obj[tly:tly + rh - 1, tlx:tlx + rw - 1]
    actual_mask = inner_mask[tly:tly + rh - 1, tlx:tlx + rw - 1]
    
    # Get the blending of the embedded area 
    sc = cv2.seamlessClone(actual_obj, dst, actual_mask, (cx, cy), clone_type)
    '''

    # OpenCV takes the bounding box of the object within the mask and places it in the coordinates
    # (cv, cy) of the destination image, this is not what we want, hence, we need to compute the
    # coordinates of the centre of the bounding box

    min_x = np.min(np.where(mask)[1])
    max_x = np.max(np.where(mask)[1])
    min_y = np.min(np.where(mask)[0])
    max_y = np.max(np.where(mask)[0])

    new_cx = int(round((max_x - min_x) / 2.))
    new_cy = int(round((max_y - min_y) / 2.))

    new_mask = mask[min_y:max_y + 1, min_x:max_x + 1].copy()

    sc = cv2.seamlessClone(obj, dst, new_mask, (new_cx, new_cy), clone_type)

    return sc

# 
# @brief Laplacian pyraid blending.
#
# @param[in] obj        Foreground image.
# @param[in] mask       Binary mask for obj.
# @param[in] dst        Background image.
# @param[in] cy         Row that indicates the point in dst where the centre of obj will be 
#                       located.
# @param[in] cx         Column that indicates the point in dst where the centre of obj will be 
#                       located.
# @param[in] num_levels Number of levels in the pyramid. 
#
# @returns the blended image.
def laplacian_pyramid_blending(obj, mask, dst, cy, cx, num_levels = 5):
    assert(obj.shape[0] == mask.shape[0] and obj.shape[1] == mask.shape[1])
    assert(obj.dtype == np.uint8 and mask.dtype == np.uint8 and dst.dtype == np.uint8)

    # print('mask:', mask.shape)
    # print('obj:', obj.shape)
    # print('dst:', dst.shape)
    
    # Some variables to make things easy 
    dst_cx = cx
    dst_cy = cy
    obj_cx = mask.shape[1] // 2
    obj_cy = mask.shape[0] // 2
    dst_h = dst.shape[0] 
    dst_w = dst.shape[1] 
    obj_h = obj.shape[0] 
    obj_w = obj.shape[1] 
    
    # Get window from dst
    top_dst = max(0, dst_cy - obj_cy)
    bottom_dst = min(dst_h, dst_cy + obj_cy) 
    left_dst = max(0, dst_cx - obj_cx)
    right_dst = min(dst_w, dst_cx + obj_cx)

    # Get window from obj
    top_obj = obj_cy - (cy - top_dst)
    bottom_obj = obj_cy + (bottom_dst - cy)
    left_obj = obj_cx - (cx - left_dst)  
    right_obj = obj_cx + (right_dst - cx)

    # Create object and destination of the same size
    new_obj = obj[top_obj:bottom_obj, left_obj:right_obj]
    new_mask = mask[top_obj:bottom_obj, left_obj:right_obj].astype(np.float32) / 255.
    new_dst = dst[top_dst:bottom_dst, left_dst:right_dst]
    assert(new_obj.shape[0] == new_mask.shape[0] and new_mask.shape[0] == new_dst.shape[0])
    assert(new_obj.shape[1] == new_mask.shape[1] and new_mask.shape[1] == new_dst.shape[1])

    # Generate Gaussian pyramid for obj, dst, and mask
    gauss_obj = new_obj.copy()
    gauss_dst = new_dst.copy()
    gauss_mask = new_mask.copy()
    gauss_pyr_obj = [gauss_obj]
    gauss_pyr_dst = [gauss_dst]
    gpM = [gauss_mask]
    for i in xrange(num_levels):
        gauss_obj = cv2.pyrDown(gauss_obj)
        gauss_dst = cv2.pyrDown(gauss_dst)
        gauss_mask = cv2.pyrDown(gauss_mask)
        gauss_pyr_obj.append(np.float32(gauss_obj))
        gauss_pyr_dst.append(np.float32(gauss_dst))
        gpM.append(np.float32(gauss_mask))

    # Generate Laplacian Pyramids for A, B and masks
    # The bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpA  = [gauss_pyr_obj[num_levels - 1]] 
    lpB  = [gauss_pyr_dst[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in xrange(num_levels - 1, 0, -1):

        sizeA = (gauss_pyr_obj[i - 1].shape[1], gauss_pyr_obj[i - 1].shape[0])
        sizeB = (gauss_pyr_dst[i - 1].shape[1], gauss_pyr_dst[i - 1].shape[0])

        # Laplacian: subtarct upscaled version of lower level from current level 
        # to get the high frequencies
        lap_obj = np.subtract(gauss_pyr_obj[i - 1], cv2.pyrUp(gauss_pyr_obj[i], dstsize = sizeA))
        lap_dst = np.subtract(gauss_pyr_dst[i - 1], cv2.pyrUp(gauss_pyr_dst[i], dstsize = sizeB))
        lpA.append(lap_obj)
        lpB.append(lap_dst)
        gpMr.append(gpM[i - 1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = np.empty_like(la) 
        ls[:,:, 0] = la[:,:, 0] * gm + lb[:,:, 0] * (1.0 - gm)
        ls[:,:, 1] = la[:,:, 1] * gm + lb[:,:, 1] * (1.0 - gm)
        ls[:,:, 2] = la[:,:, 2] * gm + lb[:,:, 2] * (1.0 - gm)
        LS.append(ls)

    # Now reconstruct
    ls_ = LS[0]
    for i in xrange(1, num_levels):
        ls_ = cv2.pyrUp(ls_, dstsize = (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i].astype(np.float32))

    # Clip those values that go out of range
    ls_ = np.clip(ls_, 0.0, 255.0)

    # Remove remaining green stuff
    kernel = np.ones((3, 3), np.uint8)
    new_mask = cv2.erode(new_mask, kernel, iterations = 1)
    new_mask = cv2.GaussianBlur(new_mask, (3, 3), 0)
    ls_[:,:, 0] = new_mask * ls_[:,:, 0] + (1 - new_mask) * new_dst[:,:, 0]
    ls_[:,:, 1] = new_mask * ls_[:,:, 1] + (1 - new_mask) * new_dst[:,:, 1]
    ls_[:,:, 2] = new_mask * ls_[:,:, 2] + (1 - new_mask) * new_dst[:,:, 2]
    
    # Put blending back on place
    result = dst.copy()
    result[top_dst:bottom_dst, left_dst:right_dst] = ls_.astype(np.uint8)

    return result

# 
# @brief Generic blending function.
#
# @param[in] obj           Foreground image.
# @param[in] mask          Binary mask for obj.
# @param[in] dst           Background image.
# @param[in] cy            Row that indicates the point in dst where the centre of obj will be 
#                          located.
# @param[in] cx            Column that indicates the point in dst where the centre of obj will 
#                          be located.
# @param[in] blending_type String indicating...
#
# @returns the result of blending obj onto dst.     
def blend(obj, mask, dst, cy, cx, blending_type):
    assert(cy >= 0 and cx >= 0 and cy < dst.shape[0] and cx < dst.shape[1])
    assert(obj.shape[0] == mask.shape[0] and obj.shape[1] == mask.shape[1])
    return BLENDING_TYPE[blending_type](obj, mask, dst, cy, cx)

BLENDING_TYPE = { 
    'alpha': alpha_blending, 
    'gaussian': gaussian_blending,
    'blood': blood_blending,
    'poisson_normal': poisson_normal_blending,
    'poisson_mixed': poisson_mixed_blending,
    'hard_light': hard_light_blending,
    'soft_light': soft_light_blending,
    'lighten_only': lighthen_only_blending,
    'dodge': dodge_blending,
    'addition': addition_blending,
    'darken_only': darken_only_blending,
    'multiply': multiply_blending,
    'difference': difference_blending,
    'subtract': subtract_blending,
    'grain_extract': grain_extract_blending,
    'grain_merge': grain_merge_blending,
    'divide': divide_blending,
    'laplacian': laplacian_pyramid_blending,
}

if __name__ == "__main__":
    raise RuntimeError('[ERROR] This is a blending module, not supposed to run as a script.')
