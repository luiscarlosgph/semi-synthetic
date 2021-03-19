"""
@brief   Module to segment chroma key images.
@author  Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
@date    26 February 2019.
"""

import os
import argparse
import cv2
import numpy as np
import multiprocessing as mp
import ntpath
import tempfile

# My imports
import common
import grabcut
import endo


class Segmenter(object):
    """@class Segmenter of objects over a chroma key."""

    def __init__(self, min_hsv_thresh=[35, 70, 15], max_hsv_thresh=[95, 255, 255],
            deinterlace=False, denoise=False, use_grabcut=True, grabcut_maxiter=5,
            grabcut_gamma=10, endo_padding=False, num_inst=None, debug=False):
        """
        @details The min_hsv_thresh and max_hsv_thresh can be a list of lists, in case that you
                 want to capture several colour ranges.
        @param[in]  min_hsv_thresh  List with the lower bounds of the HSV thresholds that captures 
                                    the chroma key, the order is [H, S, V]. The default value is
                                    [35, 70, 15], which works for green screens.
        @param[in]  max_hsv_thresh  Same as above but with the upper bounds of the HSV threshold.
                                    The default value is [95, 255, 255], which works for green
                                    screens.
        """
        self._min_hsv_thresh = min_hsv_thresh
        self._max_hsv_thresh = max_hsv_thresh 
        self.deinterlace = deinterlace
        self.denoise = denoise
        self.use_grabcut = use_grabcut
        self.grabcut_maxiter = grabcut_maxiter
        self.grabcut_gamma = grabcut_gamma
        self.endo_padding = endo_padding
        self.num_inst=num_inst
        self.debug = debug

    @staticmethod
    def deinterlace(im):
        """
        @brief This method deinterlaces an image using ffmpeg.
        @param[in]  im  Numpy ndarray image. Shape (h, w, 3) or (h, w).
        @returns the deinterlaced image.
        """
        interlaced_path = os.path.join(tempfile.gettempdir(), '.interlaced.png')
        deinterlaced_path = os.path.join(tempfile.gettempdir(), '.deinterlaced.png')
        
        # Save image in a temporary folder
        cv2.imwrite(interlaced_path, im)
        
        # Deinterlace using ffmpeg
        common.shell('ffmpeg -i ' + interlaced_path + ' -vf yadif ' + deinterlaced_path)

        # Read deinterlaced image
        deinterlaced = cv2.imread(deinterlaced_path, cv2.IMREAD_UNCHANGED)

        # Remove image from temporary folder
        common.rm(interlaced_path)
        common.rm(deinterlaced_path)

        return deinterlaced
    
    @staticmethod
    def denoise(im, median_ksize=21, gaussian_ksize=5):
        """
        @brief Denoise image to make colours more homogeneous.
        @param[in]  im              BGR input image.
        @param[in]  median_ksize    Kernel size for the median filter.
        @param[in]  gaussian_ksize  Kernel size for the Gaussian filter.
        @returns a new filtered image.
        """
        denoised = cv2.medianBlur(im, median_ksize)
        #denoised = cv2.GaussianBlur(denoised, (gaussian_ksize, gaussian_ksize), cv2.BORDER_DEFAULT)
        return denoised
    
    @staticmethod
    def hsv_bg_remove(im, min_hsv_thresh, max_hsv_thresh):
        """
        @param[in]  min_hsv_thresh  List [H, S, V] or list of lists [[H, S, V], [H, S, V]] that
                                    contain the lower bound of the threshold.
        @param[in]  max_hsv_thresh  Same as above, but indicating the upper bound of the threshold.
        @reurns a binary mask with the chroma key labeled as zero, and the rest as 255.
        """
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        
        # If the user specified just a list, we make it a list of lists
        if type(min_hsv_thresh[0]) != list:
            min_hsv_thresh = [min_hsv_thresh]
            max_hsv_thresh = [max_hsv_thresh]
        
        # Perform a union of all the chroma key colour chunks
        assert(len(min_hsv_thresh) == len(max_hsv_thresh))
        for mi, ma in zip(min_hsv_thresh, max_hsv_thresh):
            lower_bound = (mi[0], mi[1], mi[2])
            upper_bound = (ma[0], ma[1], ma[2])
            new_mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.bitwise_or(mask, new_mask)

        # We actually want a mask of the foreground
        mask = 255 - mask

        return mask

    @staticmethod
    def fill_connected_components(mask): 
        """
        @param[in]  mask  np.ndarray of a mask with holes.
        @returns a new mask with filled holes.
        """
        not_mask = cv2.bitwise_not(mask)
        contour, hier = cv2.findContours(not_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(not_mask, [cnt], 0, 255, -1)
        new_mask = cv2.bitwise_not(not_mask)
        return new_mask
    
    @staticmethod
    def clahe(im):
        """@returns the same image but with enhanced local contrast."""
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        new_im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return new_im

    @staticmethod
    def correct_lighting(im):
        height, width = im.shape[:2]

        # Generating vignette mask using Gaussian kernels
        kernel_x = cv2.getGaussianKernel(width, 150)
        kernel_y = cv2.getGaussianKernel(height, 150)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)

        new_im = im.copy()
        for i in range(3):
            new_im[:, :, i] = new_im[:, :, i] / mask    

        hsv = cv2.cvtColor(new_im, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3 
        hsv[:, :, 1][hsv[:, :, 1] > 255]  = 255
        hsv[:, :, 2] = hsv[:, :, 2] * 1.3 
        hsv[:, :, 2][hsv[:, :, 2] > 255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        new_im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return new_im

    @staticmethod
    def convex_hull(mask):
        """@returns the convex hull of a binary mask."""
        new_mask = np.zeros_like(mask)

        # Find the contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # For each contour, find the convex hull and draw it
        # on the original image.
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(new_mask, [hull], -1, 255, -1)

        return new_mask 

    @staticmethod
    def polycontour(mask, epsilon=0.001):
        """
        @param[in]  mask  Input segmentation binary mask (np.ndarray).
        @param[in]  epsilon  Smaller epsilon makes the polinomial closer to the original contour.
        @returns a mask whose contour has been fitted by a polinomial.
        """
        new_mask = np.zeros_like(mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Find the contour of largest area
            cnt = contours[0]
            max_area = cv2.contourArea(cnt)
            for cont in contours:
                if cv2.contourArea(cont) > max_area:
                    cnt = cont
                    max_area = cv2.contourArea(cont)
            
            # Draw contour using a polynomial approximation
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon * perimeter, True)
            cv2.drawContours(new_mask, [approx], -1, 255, -1)
        return new_mask

    @staticmethod
    def smooth_contour(mask, ks=5):
        """
        @param[in]  mask  Input segmentation binary mask (np.ndarray).
        @param[in]  ks    Smoothing kernel size.
        @returns a new binary mask with smoother contours.
        """
        smooth = cv2.pyrUp(mask)
        for i in range(15):
            smooth = cv2.medianBlur(smooth, ks)
        smooth = cv2.pyrDown(smooth) 
        _, smooth = cv2.threshold(smooth, 200, 255, cv2.THRESH_BINARY)
        return smooth

    @staticmethod
    def dilate(mask, ks=5, iterations=1):
        kernel = np.ones((ks, ks), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        return dilated_mask

    def segment(self, im):
        """
        @brief Binary segmentation of the objects on top of the chroma key.
        @param[in]  im  BGR image to be segmented.
        @returns a numpy.ndarray of shape (h, w) and type np.uint8 with a label of 0 for the
                 chroma key and 255 for the foreground objects.
        """
        # Deinterlace
        if self.deinterlace:
            im = Segmenter.deinterlace(im)

        # Denoise
        denoised_im = Segmenter.denoise(im) if self.denoise else im
        
        # Get HSV-based segmentation mask
        mask = Segmenter.hsv_bg_remove(denoised_im, self.min_hsv_thresh, self.max_hsv_thresh)
        
        # Mark the pixels outside the endoscopic view as background
        endomask = None
        if self.endo_padding:
            endoseg = endo.Segmenter()
            endomask = endoseg.segment(denoised_im)
            mask = cv2.bitwise_and(mask, endomask)
        
        # Grabcut post-processing
        if self.use_grabcut:
            # Preprocess mask for GrabCut marking background pixels as 'unknown'
            mask[mask == 0] = 128

            # The pixels outside the endoscopic area are set to sure background
            if endomask is not None:
                mask[endomask == 0] = 0

            # GrabCut segmentation
            bgra_im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            gc = grabcut.GrabCut(self.grabcut_maxiter)
            grabcut_mask = gc.estimateSegmentationFromTrimap(bgra_im, mask, self.grabcut_gamma)
            mask = 255 * grabcut_mask

        # If the user specifies the number of instruments, we only consider the 
        # self.num_inst largest connected components
        if self.num_inst is not None:
            mask = endo.Segmenter.largest_cc_mask(mask, ncc=self.num_inst)

        return mask

    def _segment_and_save_worker(self, inputf, outputf):
        im = cv2.imread(inputf)
        mask = self.segment(im)
        cv2.imwrite(outputf, mask)

    def segment_and_save(self, input_files, output_files):
        """
        @brief Segment a list of images.
        @param[in]  input_files   List of paths to the input images.
        @param[in]  output_files  List of paths where we should save the output binary segmentation 
                                  images.
        @returns nothing.
        """
        if self.debug:
            for inputf, outputf in zip(input_files, output_files):
                self._segment_and_save_worker(inputf, outputf)
        else:
            pool = mp.Pool()
            for inputf, outputf in zip(input_files, output_files):
                pool.apply_async(self._segment_and_save_worker, args=(inputf, outputf))
            pool.close()
            pool.join()

    @property
    def min_hsv_thresh(self):
        return self._min_hsv_thresh
    
    @min_hsv_thresh.setter
    def min_hsv_thresh(self, min_hsv_thresh):
        self._min_hsv_thresh = min_hsv_thresh
    
    @property
    def max_hsv_thresh(self):
        return self._max_hsv_thresh
    
    @max_hsv_thresh.setter
    def max_hsv_thresh(self, max_hsv_thresh):
        self._max_hsv_thresh = max_hsv_thresh


def parse_command_line_parameters(parser):      
    """
    @brief  Parses the command line parameters provided by the user and makes sure that mandatory
            parameters are present.
    @param[in]  parser  argparse.ArgumentParser
    @returns an object with the parsed arguments. 
    """
    msg = {
        '--input-dir':            'Path to the input folder.',
        '--output-dir':           'Path to the output folder.',
        '--centre-crop':          """Crop the centre of the image. The value is the proportion of 
                                     the image that will be cropped, e.g. 0.5 means crop and produce 
                                     and image half the size of orginal one.""",
        '--num-inst':             'Number of instruments.', 
        '--seg-suffix':           'Suffix of the files containing a segmentation.',
        '--min-hsv-thresh':       """Tuple with minimum bound for HSV threshold. Syntax: (H, S, V).
                                     Hue range is [0, 179], saturation range is [0, 255], and value 
                                     range is [0, 255].""",
        '--max-hsv-thresh':       """Tuple with maximum bound for HSV threshold. Syntax: (H, S, V). 
                                     Hue range is [0, 179], saturation range is [0, 255], and value 
                                     range is [0, 255].""",
        '--grabcut':              'Set it to 1 to activate the use of Grabcut.',
        '--deinterlace':          'Deinterlace the images as a preprocessing step.',
        '--refine-existing-mask': """Set it to 1 to use the existing mask as a scribble for the 
                                     segmentation.""",
        '--denoise':              'Set it to one to denoise the image before the HSV segmentation.',
        '--debug':                """Set it to 1 to go in debug mode, i.e. no parallel image 
                                     generation.""",
        '--endo-padding':         'Set it to 1 to ignore the black endoscopic padding.',
    }

    # Mandatory parameters
    parser.add_argument('--input-dir', required=True, help=msg['--input-dir'])
    parser.add_argument('--output-dir', required=True, help=msg['--output-dir'])
    
    # Optional parameters 
    parser.add_argument('--centre-crop', required=False, default=1.0, help=msg['--centre-crop'])
    parser.add_argument('--num-inst', required=False, default=None, help=msg['--num-inst'])
    parser.add_argument('--seg-suffix', required=False, default='_seg', 
        help=msg['--seg-suffix'])
    parser.add_argument('--min-hsv-thresh', required=False, default='(35, 70, 15)', 
        help=msg['--min-hsv-thresh'])
    parser.add_argument('--max-hsv-thresh', required=False, default='(95, 255, 255)', 
        help=msg['--max-hsv-thresh'])
    parser.add_argument('--grabcut', required=False, default=False, help=msg['--grabcut'])
    parser.add_argument('--deinterlace', required=False, default=False, help=msg['--deinterlace'])
    parser.add_argument('--refine-existing-mask', required=False, default=False, 
        help=msg['--refine-existing-mask'])
    parser.add_argument('--denoise', required=False, default=False, help=msg['--denoise'])
    parser.add_argument('--debug', required=False, default=False, help=msg['--debug'])
    parser.add_argument('--endo-padding', required=False, default=False, help=msg['--endo-padding'])
    
    # Parse command line
    args = parser.parse_args()

    return args


def validate_cmd_param(args):
    """
    @brief  The purpose of this function is to assert that the parameters passed in the 
            command line are ok.
    @param[in]  args  Parsed command line parameters.
    @returns nothing.
    """
    if not common.dir_exists(args.input_dir):
        raise ValueError('[validate_cmd_param] Error, the input folder does not exist.')
    if common.dir_exists(args.output_dir):
        raise ValueError('[validate_cmd_param] Error, the output folder already exists.')
    assert(type(eval(args.min_hsv_thresh) == tuple))
    assert(type(eval(args.max_hsv_thresh) == tuple))
    args.min_hsv_thresh = eval("'" + args.min_hsv_thresh + "'")
    args.max_hsv_thresh = eval("'" + args.max_hsv_thresh + "'")
    assert(int(args.grabcut) == 0 or int(args.grabcut) == 1)
    assert(int(args.deinterlace) == 0 or int(args.deinterlace) == 1)
    assert(int(args.refine_existing_mask) == 0 or int(args.refine_existing_mask) == 1) 
    if args.num_inst is not None:
        assert(int(args.num_inst) > 0)
    assert(int(args.denoise) == 0 or int(args.denoise) == 1)
    assert(float(args.centre_crop) >= 0.0)
    assert(bool(int(args.debug)) == False or bool(int(args.debug)) == True)
    assert(bool(int(args.endo_padding)) == False or bool(int(args.endo_padding)) == True)


def convert_args_to_correct_datatypes(args):
    """
    @brief  Convert the parameter strings to the right datatypes.
    @param[in,out]  args  Parsed command line parameters.
    @returns nothing.
    """
    args.min_hsv_thresh = eval(args.min_hsv_thresh)
    args.max_hsv_thresh = eval(args.max_hsv_thresh)
    args.deinterlace = bool(int(args.deinterlace))
    args.grabcut = bool(int(args.grabcut))
    if args.num_inst is not None:
        args.num_inst = int(args.num_inst)
    args.denoise = bool(int(args.denoise))
    args.refine_existing_mask = bool(int(args.refine_existing_mask))
    args.centre_crop = float(args.centre_crop)
    args.debug = bool(int(args.debug))
    args.endo_padding = bool(int(args.endo_padding))


def build_prior_list(image_list, seg_suffix, seg_ext='.png'):
    """@returns a list of paths with the segmentation priors of the provided list of images."""
    prior_list = []
    for ifile in image_list:
        fdir = os.path.dirname(ifile) 
        fname = ntpath.basename(ifile)    
        name, ext = os.path.splitext(fname)
        prior_fname = name + seg_suffix + seg_ext
        prior_path = os.path.join(fdir, prior_fname) 
        if common.file_exists(prior_path):
            prior_list.append(prior_path)
        else:
            prior_list.append(None)
    return prior_list


def build_output_list(image_list, output_dir, seg_suffix, seg_ext='.png'):
    """@returns a list of paths where the output segmentations should be stored."""
    output_list = []
    for ifile in image_list:
        fdir = os.path.dirname(ifile) 
        fname = ntpath.basename(ifile)    
        name, ext = os.path.splitext(fname)
        output_fname = name + seg_suffix + seg_ext
        output_path = os.path.join(output_dir, output_fname)
        output_list.append(output_path)
    return output_list


def main(): 
    # Process command line parameters
    common.write_info('Reading command line parameters... ')
    parser = argparse.ArgumentParser()
    args = parse_command_line_parameters(parser)
    validate_cmd_param(args)
    convert_args_to_correct_datatypes(args)
    common.writeln_ok()

    # Get list of images to be segmented
    input_list = common.listdir_absolute_no_hidden(args.input_dir)
    image_list = [fname for fname in input_list if args.seg_suffix not in fname]
    common.writeln_info('Number of images in input folder: ' + str(len(image_list)))
    prior_list = build_prior_list(image_list, args.seg_suffix)

    # Create output directory and list of output images
    common.mkdir(args.output_dir)
    seg_list = build_output_list(image_list, args.output_dir, args.seg_suffix)
    
    # Segment all the images and store the results in the output folder
    segmenter = Segmenter(min_hsv_thresh=args.min_hsv_thresh, 
        max_hsv_thresh=args.max_hsv_thresh, deinterlace=args.deinterlace, denoise=args.denoise,
        use_grabcut=args.grabcut, endo_padding=args.endo_padding, num_inst=args.num_inst,
        debug=args.debug)
    segmenter.segment_and_save(image_list, seg_list)

    actual_output_list = common.listdir_absolute_no_hidden(args.output_dir)
    common.writeln_info('Number of segmentations in output folder: ' + str(len(actual_output_list)))
    

if __name__ == '__main__':
    main()
