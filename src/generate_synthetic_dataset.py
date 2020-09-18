"""
@brief   This script generates a synthetic dataset for tool-background segmentation by blending
         the foregrounds on the backgrounds. The folder that contains foregrounds must also have
         segmentation masks.
@author  Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
@date    11 Mar 2019.
"""

import argparse
import cv2
import numpy as np
import re
import multiprocessing as mp
import os
import scipy.stats
import time

# My imports
import common
import blending
import image_generator


def parse_command_line_parameters(parser):
    """
    @brief  Parses the command line parameters provided by the user.
    @param[in]  parser  argparse.ArgumentParser
    @returns an object with the parsed arguments that can be used as a dictionary.
    """
    parser.add_argument(
        '--input-fg-dir',
        required=True,
        help='Path to the input folder with foreground images.',
    )
    parser.add_argument(
        '--input-bg-dir',
        required=True,
        help='Path to the input folder with background images.'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Path to the output folder.',
    )
    parser.add_argument(
        '--gt-suffix',
        required=True,
        help='Suffix of the ground truth files.',
    )
    parser.add_argument(
        '--classes',
        required=True,
        help='Number of classes in the segmentation masks.'
    )
    parser.add_argument(
        '--class-map',
        required=True,
        help='JSON file with the mapping from intensity to class id. Used for sanity checks.'
    )
    parser.add_argument(
        '--blend-modes',
        required=True,
        help='List in Python style with the names of the blending types to be used. '
            + 'Candidates are: ' + str(blending.BLENDING_TYPE.keys())
    )
    parser.add_argument(
        '--num-samples',
        required=True,
        help='Number of samples to generate.'
    )
    parser.add_argument(
        '--width',
        required=True,
        help='Standardise the width of the output images.'
    )

    # Optional parameters
    parser.add_argument(
        '--debug',
        required=False,
        default=False,
        help='The output folder is populated with the images generated along the pipeline.'
    )
    parser.add_argument(
        '--processes',
        required=False,
        default=mp.cpu_count(),
        help='Number of processes to run in parallel.'
    )
    parser.add_argument(
        '--first-image-id',
        required=False,
        default=0,
        help='Number of the first image in the generated dataset. Default is zero.'
    )
    parser.add_argument(
        '--fg-aug',
        required=False,
        default='{}',
        help='Dictionary of foreground augmentation methods to be \
                applied to the foreground.'
    )
    parser.add_argument(
        '--bg-aug',
        required=False,
        default='{}',
        help='Dictionary of background augmentation methods to be \
            applied to the background.'
    )
    parser.add_argument(
        '--blend-aug',
        required=False,
        default='{}',
        help='Dictionary of background augmentation methods to be \
            applied to the blended images.'
    )

    # Parse command line
    args = parser.parse_args()

    return args


def validate_cmd_param(args):
    """
    @brief The purpose of this function is to assert that the parameters passed in the command line
           are ok.
    @param[in]  args  Parsed command line parameters.
    @returns nothing.
    """
    assert(common.dir_exists(args.input_fg_dir))
    assert(common.dir_exists(args.input_bg_dir))
    if common.dir_exists(args.output_dir):
        raise ValueError('[ERROR] The output folder already exists.')
    assert(int(args.classes) > 1)
    assert(common.file_exists(args.class_map))
    for bm in eval(args.blend_modes):
        assert(bm in blending.BLENDING_TYPE.keys())
    assert(int(args.num_samples) > 0)

    # If the debug flag is not false, it should contain a number passed
    # through the command line
    if args.debug:
        assert(int(args.debug) == 0 or int(args.debug) == 1)

    # Check the type of the input width for standardised images
    assert(int(args.width) > 0)
    assert(int(args.processes) > 0)
    assert(int(args.first_image_id) >= 0)
    assert(isinstance(eval(args.fg_aug), dict))
    assert(isinstance(eval(args.bg_aug), dict))
    assert(isinstance(eval(args.blend_aug), dict))

    # Check that the foreground or background augmentations are known by the
    # image_generator
    be_dict = image_generator.AugmentationEngine.BACKENDS
    accepted_options = []
    for key in be_dict:
        accepted_options += be_dict[key]
    for opt in eval(args.bg_aug):
        if opt not in accepted_options:
            raise ValueError(
                'The augmentation option "' +
                opt +
                '" is unknown.')
    for opt in eval(args.fg_aug):
        if opt not in accepted_options:
            raise ValueError(
                'The augmentation option "' +
                opt +
                '" is unknown.')


def convert_args_to_correct_datatypes(args):
    """
    @brief  Convert the parameter strings to the right datatypes.
    @param[in,out]  args  Parsed command line parameters.
    @returns nothing.
    """
    args.classes = int(args.classes)
    args.class_map = common.convert_dict_to_int(
        common.load_json_as_dict(args.class_map))
    args.blend_modes = eval(args.blend_modes)
    args.num_samples = int(args.num_samples)
    args.debug = bool(int(args.debug))
    args.width = int(args.width)
    args.processes = int(args.processes)
    args.first_image_id = int(args.first_image_id)
    args.bg_aug = eval(args.bg_aug)
    args.fg_aug = eval(args.fg_aug)
    args.blend_aug = eval(args.blend_aug)


def save_image(dir_path, file_name, im,
               flags=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
    """
    @brief Function to save an image to the output directory.
           Typically for debugging purposes.
    """
    fpath = os.path.join(dir_path, file_name)
    cv2.imwrite(fpath, im.astype(np.uint8))


def worker(gen_count, fg_list, bg_list, args, max_ninst=3):
    """
    @brief Forkable main.
    @param[in]  gen_count  Index of the image to generate.
    @param[in]  fg_list    List of foreground image paths.
    @param[in]  bg_list    List of background image paths.
    @param[in]  args       Command line arguments.
    @returns nothing.
    """
    # Give each subprocess a different seed so they do not generate the same images
    # random.seed(gen_count)
    np.random.seed(gen_count)

    # Create synthetic image generator
    im_gen = image_generator.ImageGenerator(
        bg_list, fg_list, bg_aug=args.bg_aug,
        fg_aug=args.fg_aug, blend_aug=args.blend_aug)

    # Generate a synthetic image blended with different methods
    fname = ("%012d" % gen_count)
    blended_images = im_gen.generate_image(args.blend_modes, width=args.width)
    for mode in blended_images:
        im, mask = blended_images[mode]
        save_image(args.output_dir, fname + '_' + mode + '.jpg', im)
        save_image(
            args.output_dir,
            fname +
            '_' +
            mode +
            args.gt_suffix +
            '.png',
            mask)
        common.writeln_info('Image ' + fname + ' (' + mode + ')' + ' saved.')


def main():
    # Process command line parameters
    common.write_info('Reading command line parameters... ')
    parser = argparse.ArgumentParser()
    args = parse_command_line_parameters(parser)
    validate_cmd_param(args)
    convert_args_to_correct_datatypes(args)
    common.writeln_ok()

    # Make sure we do not use the same seed across executions: in case we want to run several
    # scripts and then put the data together
    # random.seed()
    np.random.seed()

    # Create output folder and initialise generator counter
    regex = re.compile(args.gt_suffix + '[.]...$')
    if not common.dir_exists(args.output_dir):
        common.mkdir(args.output_dir)
        gen_count = 0
    else:
        tmp_list = common.natsort(common.listdir_no_hidden(args.output_dir))
        tmp_list = [x for x in tmp_list if not regex.search(x)]
        gen_count = int(common.get_fname_no_ext(tmp_list[-1])) + 1

    # Create lists of foreground and background
    fg_list = common.listdir_absolute_no_hidden(args.input_fg_dir)
    fg_list = [x for x in fg_list if not regex.search(x)]
    bg_list = common.listdir_absolute_no_hidden(args.input_bg_dir)

    # Do not use multiprocessing module if debug mode is selected
    tic = time.time()
    if args.debug:
        # Run data generation sequentially
        for i in range(args.first_image_id, args.first_image_id + args.num_samples):
            worker(i, fg_list, bg_list, args)
    else:
        # Generate all images in parallel
        pool = mp.Pool(args.processes, maxtasksperchild=1)
        for i in range(args.first_image_id, args.first_image_id + args.num_samples):
            pool.apply_async(worker, args=(i, fg_list, bg_list, args))
        pool.close()
        pool.join()
    toc = time.time()
    elap = toc - tic
    common.writeln_info(str(args.num_samples) +
                        ' images generated in ' + str(elap) + ' seconds.')

if __name__ == "__main__":
    main()
