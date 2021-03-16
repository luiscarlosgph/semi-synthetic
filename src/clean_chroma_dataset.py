##
# @brief   This script preprocesses a folder with green chroma images and cleans them.
#
# @author Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   15 March 2021.

import argparse
import cv2
import numpy as np
import multiprocessing as mp
import re

# My imports
import common
import image
import datareader
import datawriter
import chroma
import geometry


def parse_command_line_parameters(parser):      
    """
    @brief  Parses the command line parameters provided by the user and makes sure that mandatory
            parameters are present.
    @param[in]  parser  argparse.ArgumentParser
    @returns an object with the parsed arguments. 
    """
    parser.add_argument(
        '--input-dir',
        required = True,
        help = 'Path to the input folder.',
    )
    parser.add_argument(
        '--output-dir',
        required = True,
        help = 'Path to the output folder.',
    )
    parser.add_argument(
        '--centre-crop',
        required = True,
        help = 'Crop the centre of the image. The value is the proportion of the image that ' \
            + 'will be cropped, e.g. 0.5 means crop and produce and image half the size of ' \
            + 'orginal one.'
    )
    
    # Optional parameters 
    parser.add_argument(
        '--instruments',
        required = False,
        default = None,
        help = 'Number of instruments.'
    )
    parser.add_argument(
        '--gt-suffix',
        required = False,
        default = None,
        help = 'Suffix of the ground truth files.',
    )
    parser.add_argument(
        '--min-hsv-thresh',
        required = False,
        default = '(35, 70, 15)',
        help = 'Tuple with minimum bound for HSV threshold.'
    )
    parser.add_argument(
        '--max-hsv-thresh',
        required = False,
        default = '(95, 255, 255)',
        help = 'Tuple with maximum bound for HSV threshold.'
    )
    parser.add_argument(
        '--grabcut',
        required = False,
        default = False,
        help = 'Boolean to activate the use of Grabcut.'
    )
    parser.add_argument(
        '--deinterlace',
        required = False,
        default = False,
        help = 'Run algorithm to deinterlace the images as a preprocessing step.'
    )
    parser.add_argument(
        '--remove-recording-text',
        required = False,
        default = False,
        help = 'Remove red text with recording text.'
    )
    parser.add_argument(
        '--remove-hsv-text',
        required = False,
        default = False,
        help = 'Remove blue text with HSV info.'
    )
    parser.add_argument(
        '--refine-existing-mask',
        required = False,
        default = False,
        help = 'Use the existing mask as a base for the cleaning.'
    )
    parser.add_argument(
        '--erode',
        required = False,
        default = 0,
        help = 'Size of the eroding kernel, if any.'
    ) 
    parser.add_argument(
        '--debug',
        required = False,
        default = 0,
        help = 'Set it to 1 to go in debug mode, i.e. no parallel image generation.'
    )
    
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
    assert(int(args.remove_recording_text) == 0 or int(args.remove_recording_text) == 1)
    assert(int(args.remove_hsv_text) == 0 or int(args.remove_hsv_text) == 1)
    assert(int(args.refine_existing_mask) == 0 or int(args.refine_existing_mask) == 1) 
    if args.instruments is not None:
        assert(int(args.instruments) > 0)
    assert(int(args.erode) >= 0)
    assert(float(args.centre_crop) >= 0.0)
    assert(bool(int(args.debug)) == False or bool(int(args.debug)) == True)


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
    args.remove_recording_text = bool(int(args.remove_recording_text))
    args.remove_hsv_text = bool(int(args.remove_hsv_text))
    if args.instruments is not None:
        args.instruments = int(args.instruments)
    args.erode = int(args.erode)
    args.refine_existing_mask = bool(int(args.refine_existing_mask))
    args.centre_crop = float(args.centre_crop)
    args.debug = bool(int(args.debug))


def multiprocessing_func(path, args):
    # Read file and/or mask
    fdir = common.get_file_dir(path)
    fname = common.get_fname_no_ext(path)
    fext = common.get_ext(path)
    read_img = image.CaffeinatedImage.from_file(path, fname + fext)
    read_gt = None 
    if args.gt_suffix:
        gt_path = fdir + '/' + fname + args.gt_suffix + fext
        gt_fname = common.get_fname_no_ext(gt_path)
        gt = image.CaffeinatedLabel.from_file(gt_path, gt_fname + fext, args.classes,
            args.class_map)
    
    if read_img and read_gt:
        common.writeln_info('Processing ' + read_img.name + ' ... ')
         
        # Warn the user if the mask is not a single-channel image
        if len(read_gt.raw.shape) > 2:
            common.writeln_warn('The ground truth \'' + read_gt.name \
                + '\' is not single-channel!')

        # Deinterlace the image
        if args.deinterlace:
            read_img._raw_frame = image.CaffeinatedImage.deinterlace(
                read_img._raw_frame) 

        # Mask recording text
        if args.remove_recording_text:
            read_img_hsv = cv2.cvtColor(read_img.raw, cv2.COLOR_BGR2HSV)
            mask_recording = cv2.inRange(read_img_hsv, (0, 100, 100), 
                (15, 255, 255))
            mask_recording[0:-35, :] = 0 
            mask_recording[-10:, :] = 0 
            mask_recording[:, 120:] = 0
            mask_recording[:, :5] = 0
        else:
            mask_recording = read_gt.raw

        # Mask HSV info
        if args.remove_hsv_text:
            read_img_hsv = cv2.cvtColor(read_img.raw, cv2.COLOR_BGR2HSV)
            mask_hsv_info = cv2.inRange(read_img_hsv, (110, 150, 150), 
                (130, 255, 255))
            mask_hsv_info[90:, :] = 0
            mask_hsv_info[:, 150:] = 0
        else:
            mask_hsv_info = read_gt.raw
        
        # Join masks and dilate a bit to make sure the text is covered
        if args.remove_recording_text or args.remove_hsv_text:
            noise_mask = cv2.bitwise_or(mask_recording, mask_hsv_info)
            kernel = np.ones((3, 3), np.uint8)
            noise_mask = cv2.dilate(noise_mask, kernel, iterations = 1)

            # Inpaint the image to remove the text
            frame = cv2.inpaint(read_img.raw, noise_mask, 3, cv2.INPAINT_TELEA)
        else:
            frame = read_img.raw
        
        # Refine mask if the user wants (logical and between the previous one and
        # the new one)
        hsv_lb = args.min_hsv_thresh
        hsv_ub = args.max_hsv_thresh
        if args.refine_existing_mask:
            #mod_frame = frame.copy()
            #mod_frame[:, :, 0][read_gt.raw == 0] = 0
            #mod_frame[:, :, 1][read_gt.raw == 0] = 255
            #mod_frame[:, :, 2][read_gt.raw == 0] = 0
            #mask = chroma.chroma_key_mask(mod_frame, hsv_lb[0], hsv_ub[0],
            #    hsv_lb[1], hsv_ub[1], hsv_lb[2], hsv_ub[2])
            mask = chroma.chroma_key_mask(frame, hsv_lb[0], hsv_ub[0],
                hsv_lb[1], hsv_ub[1], hsv_lb[2], hsv_ub[2])
            mask = cv2.bitwise_and(mask, read_gt.raw)
        else:
            # Filter green chroma in HSV and generate segmentation mask
            #dframe = chroma.denoise(frame, median_ksize = 15)
            mask = chroma.chroma_key_mask(frame, hsv_lb[0], hsv_ub[0],
                hsv_lb[1], hsv_ub[1], hsv_lb[2], hsv_ub[2])
        
        # Erode mask a bit to remove noise and green from the borders
        #frame = chroma.reduce_green(frame, mask, hsv_lb, hsv_ub)
        #mask = cv2.dilate(mask, kernel, iterations = 1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations = 1)
        mask = cv2.medianBlur(mask, 3)

        # Threshold mask 
        mask = image.CaffeinatedAbstract.bin_thresh(mask)

        # Grabcut with the provided mask
        if args.grabcut:
            bgd_model = np.zeros((1, 65),np.float64)
            fgd_model = np.zeros((1, 65),np.float64)
            mask[mask != 255] = cv2.GC_PR_BGD
            mask[mask == 255] = cv2.GC_PR_FGD
            mask, bgd_model, fgd_model = cv2.grabCut(frame, mask, None,
                bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            mask[mask == cv2.GC_PR_FGD] = 255
            mask[mask == cv2.GC_FGD] = 255
            mask[mask != 255] = 0
        
        # Save the frame in the destination folder
        #dw.next(image.CaffeinatedImage(frame, read_img.name))
        #dw.next(image.CaffeinatedImage(mask, read_gt.name))
        #sanity_count += 1
        common.writeln_ok()

    else:
        common.writeln_info('Processing ' + read_img.name + ' without mask ... ')
         
        # Deinterlace the image
        if args.deinterlace:
            read_img._raw_frame = image.CaffeinatedImage.deinterlace(
                read_img._raw_frame) 

        # Mask recording text
        if args.remove_recording_text:
            read_img_hsv = cv2.cvtColor(read_img.raw, cv2.COLOR_BGR2HSV)
            mask_recording = cv2.inRange(read_img_hsv, (0, 100, 100), 
                (15, 255, 255))
            mask_recording[0:-35, :] = 0 
            mask_recording[-10:, :] = 0 
            mask_recording[:, 120:] = 0
            mask_recording[:, :5] = 0
        else:
            mask_recording = None

        # Mask HSV info
        if args.remove_hsv_text:
            read_img_hsv = cv2.cvtColor(read_img.raw, cv2.COLOR_BGR2HSV)
            mask_hsv_info = cv2.inRange(read_img_hsv, (110, 150, 150), 
                (130, 255, 255))
            mask_hsv_info[90:, :] = 0
            mask_hsv_info[:, 150:] = 0
        else:
            mask_hsv_info = None
        
        # Join masks and dilate a bit to make sure the text is covered
        if args.remove_recording_text and args.remove_hsv_text:
            noise_mask = cv2.bitwise_or(mask_recording, mask_hsv_info)
            kernel = np.ones((3, 3), np.uint8)
            noise_mask = cv2.dilate(noise_mask, kernel, iterations = 1)
            frame = cv2.inpaint(read_img.raw, noise_mask, 3, cv2.INPAINT_TELEA)
        elif args.remove_recording_text:
            noise_mask = mask_recording
            kernel = np.ones((3, 3), np.uint8)
            noise_mask = cv2.dilate(noise_mask, kernel, iterations = 1)
            frame = cv2.inpaint(read_img.raw, noise_mask, 3, cv2.INPAINT_TELEA)
        elif args.remove_hsv_text:
            noise_mask = mask_hsv_info
            kernel = np.ones((3, 3), np.uint8)
            noise_mask = cv2.dilate(noise_mask, kernel, iterations = 1)
            frame = cv2.inpaint(read_img.raw, noise_mask, 3, cv2.INPAINT_TELEA)
        else:
            frame = read_img.raw

        # Crop image and mask in the centre
        new_h = int(round(frame.shape[0] * args.centre_crop))
        new_w = int(round(frame.shape[1] * args.centre_crop))
        frame = image.CaffeinatedAbstract.crop_center(frame, new_w, new_h)

        # Filter green chroma in HSV and generate segmentation mask
        #dframe = chroma.denoise(frame, median_ksize = 15)
        hsv_lb = args.min_hsv_thresh
        hsv_ub = args.max_hsv_thresh
        mask = chroma.chroma_key_mask(frame, hsv_lb[0], hsv_ub[0],
            hsv_lb[1], hsv_ub[1], hsv_lb[2], hsv_ub[2])

        # Erode mask a bit to remove noise and green from the borders
        #frame = chroma.reduce_green(frame, mask, hsv_lb, hsv_ub)
        #mask = cv2.dilate(mask, kernel, iterations = 1)
        #kernel = np.ones((3, 3), np.uint8)
        #mask = cv2.erode(mask, kernel, iterations = 1)
        #mask = cv2.medianBlur(mask, 3)

        # Threshold mask 
        mask = image.CaffeinatedAbstract.bin_thresh(mask)

        # Grabcut with the provided mask
        if args.grabcut:
            bgd_model = np.zeros((1, 65),np.float64)
            fgd_model = np.zeros((1, 65),np.float64)
            mask[mask != 255] = cv2.GC_PR_BGD
            mask[mask == 255] = cv2.GC_PR_FGD
            mask, bgd_model, fgd_model = cv2.grabCut(frame, mask, None,
                bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            mask[mask == cv2.GC_PR_FGD] = 255
            mask[mask == cv2.GC_FGD] = 255
            mask[mask != 255] = 0
        
        # Get the largest connected components
        if args.instruments:
            mask_cc = np.zeros_like(mask, dtype = np.uint8)
            cnts = cv2.findContours(mask, cv2.RETR_LIST, 
                cv2.CHAIN_APPROX_SIMPLE)[1]
            cnts_sorted_area = sorted(cnts, key = lambda x: -cv2.contourArea(x))
            accepted_cnts = cnts_sorted_area[:args.instruments]
            cv2.drawContours(mask_cc, accepted_cnts, -1, 255, -1)
            mask = cv2.bitwise_and(mask, mask_cc)

        # Erode mask to remove a bit of green
        if args.erode:
            kernel = np.ones((args.erode, args.erode), np.uint8)
            mask = cv2.erode(mask, kernel, iterations = 1)
                                
        # Save the frame in the destination folder
        #dw.next(image.CaffeinatedImage(frame, read_img.name))
        #dw.next(image.CaffeinatedImage(mask, read_img.name + '_seg.png'))
        #sanity_count += 1

        common.writeln_ok()
    
    # Write file to disk
    try:
        output_image = image.CaffeinatedImage(frame, read_img.name)
        output_image.save(args.output_dir + '/' + output_image.name, None)
        output_mask = image.CaffeinatedImage(mask, fname + '_seg.png')
        output_mask.save(args.output_dir + '/' + output_mask.name, None)
    except:
        common.writeln_error('The image ' + output_image.name + ' could not be saved.')
    

def main(): 
    # Process command line parameters
    common.write_info('Reading command line parameters... ')
    parser = argparse.ArgumentParser()
    args = parse_command_line_parameters(parser)
    validate_cmd_param(args)
    convert_args_to_correct_datatypes(args)
    common.writeln_ok()
    
    # Read list of images
    sanity_count = 0
    args.classes = 2
    args.class_map = {0: 0, 255 : 1}
    #bgd_model = np.zeros((1, 65),np.float64)
    #fgd_model = np.zeros((1, 65),np.float64)
    file_list = None 
    temp_file_list = common.listdir_absolute_no_hidden(args.input_dir)
    if args.gt_suffix:
        regex = re.compile(args.gt_suffix + '[.]...$') 
        file_list = [ x for x in temp_file_list if not regex.search(x) ]
    else:
        file_list = temp_file_list

    common.mkdir(args.output_dir)
    
    # If we are in debugging mode all the segmentations are generated sequentially
    if args.debug:
        for f in file_list:
            multiprocessing_func(f, args)
    else:
        pool = mp.Pool()
        for f in file_list:
            pool.apply_async(multiprocessing_func, args = (f, args))
        pool.close()
        pool.join()
    
    len_input_folder = len(common.listdir_no_hidden(args.input_dir))
    len_output_folder = len(common.listdir_no_hidden(args.output_dir))
    common.writeln_info('Images in the input folder: ' + str(len_input_folder))
    common.writeln_info('Images in the output folder: ' + str(len_output_folder)) 

if __name__ == "__main__":
    main()
