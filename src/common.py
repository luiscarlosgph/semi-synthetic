"""
@brief   Common functions used for prototyping methods and handling data.
@author  Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
@date    17 Mar 2017.
"""

import sys
import math
import os
import re
import random        # choice
import string        # ascii_lowercase
import ntpath        # basename
import numpy as np   # np.array
import tempfile      # tempfile.NamedTemporaryFile
import decimal
import imghdr
import shutil
import cv2
import json
import collections
import zipfile
import datetime

# -- Constants -- ##
ALLOWED_IMAGE_FORMATS = [
    'gif',
     'pbm',
     'pgm',
     'ppm',
     'tiff',
     'xbm',
     'jpeg',
     'bmp',
     'png']

# -- Regular expressions -- ##
INT_RE = r'(?:[-+]?\d+)'
FLOAT_RE = r'(?:[-+]?\d*\.\d+(?:[eE][-+]\d+)?|\d+)'
BLANKS_RE = r'(?:(?:[ \t\r\n])+)'

#
# @class for terminal colours. Use it like this:
#
#        print bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC
#
# @details Credit to @joeld:
#          http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
#


class bcolours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



class Colour:
    """@class Colour stores colours in BGR uint8 format."""
    RED = [0, 0, 255]
    GREEN = [0, 255, 0]
    BLUE = [255, 0, 0]
    YELLOW = [0, 255, 255]
    MAGENTA = [255, 0, 255]
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]


def gen_rand_str(length=8):
    """
    @brief Generates a random lowercase string.
    @param[in] length Desired length of the returned string.
    @returns a random string of the given length.
    """
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


#
# @brief Converts degrees to radians.
#
# @returns the amount of radians equivalent to the input 'degrees'.
def deg_to_rad(degrees):
    return math.pi * degrees / 180.0


#
# @brief Converts radians to degrees.
#
def rad_to_deg(r):
    return r * 180.0 / math.pi


#
# @brief Lists a directory.
#
# @param[in] path String containing the path (relative or absolute) to the folder that you want to
#                 list.
# @returns a list of files and folders inside the given path.
def listdir(path):
    return filter(None, sorted(os.listdir(path)))

#
# @brief Lists a directory removing the extension of the files and the hidden files.
#
# @param[in] path String containing the path (relative or absolute) to the folder that you want to
#                 list.
#
# @returns a list of -unhidden- files and folders inside the given path.
def listdir_no_hidden(path):
    return natsort([f for f in listdir(path) if not f.startswith('.')])


#
# @brief Lists a directory removing the extension of the files. Folder names will reimain untouched.
#
# @param[in] path String containing the path (relative or absolute) to the folder that you want to
#            list.
#
# @returns a list of files and folders inside the given path.
def listdir_no_ext(path):
    return [f.split('.')[0] for f in listdir(path)]


#
# @brief Lists a directory removing the extension of the files and the hidden files.
#
# @param[in] path String containing the path (relative or absolute) to the folder that you want to
#                 list.
#
# @returns a list of -unhidden- files (without extension) and folders inside the given path.
def listdir_no_ext_no_hidden(path):
    return [f for f in listdir_no_ext(path, suffix) if not f.startswith('.')]


#
# @brief Lists a directory.
#
# @param[in] path String containing the path (relative or absolute) to the folder that you want to
#                 list.
#
# @returns a list of absolute paths pointing to the files and folders inside the given path.
def listdir_absolute(path):
    absolute_path = os.path.abspath(path)
    return [absolute_path + '/' + fpath for fpath in listdir(path)]


#
# @brief Lists a directory without listing hidden files.
#
# @param[in] path String containing the path (relative or absolute) to the folder that you want to
#                 list.
#
# @returns a list of absolute paths pointing to the files and folders inside the given path.
def listdir_absolute_no_hidden(path):
    absolute_path = os.path.abspath(path)
    return [absolute_path + '/' + fpath for fpath in listdir_no_hidden(path)]


#
# @brief Writes a message to screen and flushes the buffers.
#
# @param[in] message is the message that will be printed to the user.
#
# @returns nothing.
def write_info(msg):
    sys.stdout.write(
        '[' +
        bcolours.OKGREEN +
     'INFO' +
     bcolours.ENDC +
     '] ' +
     msg)
    sys.stdout.flush()


#
# @brief Writes a message and a '\n' to screen and flushes the buffers.
#
# @param[in] msg is the message that will be printed to the user.
#
# @returns nothing.
def writeln_info(msg):
    sys.stdout.write(
        '[' + bcolours.OKGREEN + 'INFO' + bcolours.ENDC + '] ' + str(msg) + '\n')
    sys.stdout.flush()


#
# @brief Writes a warning message to screen and flushes the buffers.
#
# @param[in] message is the message that will be printed to the user.
#
# @returns nothing.
def write_warn(msg):
    sys.stdout.write(
        '[' +
        bcolours.WARNING +
     'WARN' +
     bcolours.ENDC +
     '] ' +
     msg)
    sys.stdout.flush()


#
# @brief Writes a warning message and a '\n' to screen and flushes the buffers.
#
# @param[in] msg is the message that will be printed to the user.
#
# @returns nothing.
def writeln_warn(msg):
    sys.stdout.write(
        '[' +
        bcolours.WARNING +
     'WARN' +
     bcolours.ENDC +
     '] ' +
     msg +
     '\n')
    sys.stdout.flush()


#
# @brief Writes an error message to screen and flushes the buffers.
#
# @param[in] message is the message that will be printed to the user.
#
# @returns nothing.
def write_error(msg):
    sys.stdout.write(
        '[' +
        bcolours.OKGREEN +
     'ERROR' +
     bcolours.ENDC +
     '] ' +
     msg)
    sys.stdout.flush()


#
# @brief Writes an error message and a '\n' to screen and flushes the buffers.
#
# @param[in] msg is the message that will be printed to the user.
#
# @returns nothing.
def writeln_error(msg):
    sys.stdout.write(
        '[' +
        bcolours.OKGREEN +
     'ERROR' +
     bcolours.ENDC +
     '] ' +
     msg +
     '\n')
    sys.stdout.flush()


#
# @brief Writes an 'OK\n' to screen and flushes buffer.
#
# @returns nothing.
def writeln_ok():
    sys.stdout.write('[' + bcolours.OKGREEN + 'OK' + bcolours.ENDC + "]\n")
    sys.stdout.flush()


#
# @brief Writes an '[FAIL]\n' to screen and flushes buffer.
#
# @returns nothing.
def writeln_fail():
    sys.stdout.write('[' + bcolours.FAIL + 'FAIL' + bcolours.ENDC + "]\n")
    sys.stdout.flush()


#
# @brief Extract a filename (without file extension) from a (relative or absolute) path.
#
# @param[in] path Path pointing to a file (can be relative or absolute).
#
# @returns the filename without extension.
def get_fname_no_ext(path):
    fname, ext = os.path.splitext(ntpath.basename(path))
    return fname


#
# @brief Obtain extension from file path. Warning: The dot '.' is also returned!
#
# @param[in] path Path pointing to a file (can be relative or absolute).
#
# @returns a string with the extension of the given path.
def get_ext(path):
    fname, ext = os.path.splitext(ntpath.basename(path))
    return ext


#
# @brief Function that reads a file.
#
# @param[in] path to the file to be read.
#
# @returns the contents of the file.
def read_file(path):
    with open(path, 'r') as myfile:
        contents = myfile.read()
    return contents


#
# @def Function to read each line of a file into a list.
#
# @param[in] path to the file to be read.
#
# @returns an array of lines.
def read_file_by_lines(path):
    with open(path) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    return content


#
# @param[in] path Path to the file whose existance you want to check.
#
# @returns true if the file exists, otherwise returns false.
def file_exists(fpath):
    return True if os.path.isfile(fpath) else False


#
# @param[in] path Path to the folder whose existance you want to check.
#
# @returns true if folder exists, otherwise returns false.
def dir_exists(dpath):
    return True if os.path.isdir(dpath) else False


#
# @param[in] path Path to a possible file or folder.
#
# @returns True if there is a file or folder that already exists in the given path.
def path_exists(path):
    return True if (file_exists(path) or dir_exists(path)) else False


#
# @brief Get the path to the directory that contains a certain file.
#
# @param[in] path Path to the file.
#
# @returns absolute path to the directory where the file is located.
def get_file_dir(path):
    assert(file_exists(path))
    return os.path.dirname(os.path.realpath(path))


#
# @brief Appends a string to a text file.
#
# @param[in] path Path to the text file. It should exist already.
# @param[in] text Text to be appended to the file.
#
# @returns nothing.
def append(path, text):
    # Sanity check: the file must exist.
    if not file_exists(path):
        raise RuntimeError('[append] Error the file must already exist.')

    # Append text to existing file
    with open(path, "a") as myfile:
        myfile.write(text)


#
# @brief Copies a file to a destination raising an exception if the destination file already exists.
#
# @details This function will raise exceptions if either of the files do not exist or if they are
#          not files.
#
# @param[in] source      Source path.
# @param[in] destination Destination path.
#
# @returns nothing.
def copy_file(source, destination):
    # Sanity check: source exists and it is a file
    if not file_exists(source):
        raise RuntimeError(
            '[copy_file] Error, source file [ ' +
            source +
            ' ] does not exist.')

    # Sanity check: destination file does not exist
    if file_exists(destination):
        raise RuntimeError('[copy_file] Error, destination file [ ' + destination + ' ] already '
                           + 'exists.')

    shutil.copy(source, destination)


#
# @brief Moves a file to a destination raising an exception if the destination file already exists.
#
# @details This function will raise exceptions if either of the files do not exist or if they are
#          not files.
#
# @param[in] source      Source path.
# @param[in] destination Destination path.
#
# @returns nothing.
def move_file(source, destination):
    # Sanity check: source exists and it is a file
    if not file_exists(source):
        raise RuntimeError(
            '[copy_file] Error, source file [ ' +
            source +
            ' ] does not exist.')

    # Sanity check: destination file does not exist
    if file_exists(destination):
        raise RuntimeError('[copy_file] Error, destination file [ ' + destination + ' ] already '
                           + 'exists.')

    shutil.move(source, destination)


#
# @brief Copy directory and all its contents.
#
# @param[in] src Source path to the directory.
# @param[in] dst Destination path to the directory.
#
# @returns nothing.
def copy_dir(src, dst):
    assert(path_exists(src))
    assert(not path_exists(dst))
    try:
        shutil.copytree(src, dest)
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    except OSError as e:
        print('Directory not copied. Error: %s' % e)


#
# @brief Move directory with all its contents to a new path.
#
# @param[in] src Source directory path.
# @param[in] dst Destination directory path.
#
# @returns nothing.
def move_dir(src, dst):
    assert(path_exists(src))
    assert(not path_exists(dst))
    try:
        shutil.move(src, dest)
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    except OSError as e:
        print('Directory not copied. Error: %s' % e)


#
# @brief Create a temporary directory.
#
# @returns the path to the temporary directory created.
def mk_temp_dir():
    dir_path = tempfile.gettempdir() + '/' + gen_rand_str()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        raise RuntimeError('[mk_temp_dir] Error, the randomly generated temporary directory '
                           + 'already exists.')
    return dir_path


#
# @brief Create folder.
#
# @param[in] path to the new folder.
#
# @returns nothing.
def mkdir(path):
    if os.path.exists(path):
        raise RuntimeError(
            '[mkdir] Error, this path already exists so a folder cannot be created.')
    os.makedirs(path)


#
# @brief Remove directory.
#
# @param[in] path to the folder to be deleted.
#
# @returns nothing.
def rmdir(path):
    shutil.rmtree(path)


#
# @brief Remove file. Only files can be removed with this function.
#
# @param[in] path to the file that will be removed.
#
# @returns nothing.
def rm(path):
    # Assert that the path is a file
    if not file_exists(path):
        raise RuntimeError('[rm] The given path is not a file.')

    os.unlink(path)


#
# @brief Runs a shell command and returns the output.
#
# @param[in] cmd String with the command that will be run.
#
# @returns the output (stdout and stderr) of the command.
def shell(cmd):
    # Generate the name of the random log file that will store the command
    # output
    tmp_log_file = tempfile.gettempdir() + '/' + gen_rand_str() + '.log'
    cmd += ' > ' + tmp_log_file + ' 2>&1'

    # Run command
    os.system(cmd)

    # Read command output from logfile
    fd = open(tmp_log_file, 'r')
    output = fd.read()
    fd.close()

    # Delete log file
    rm(tmp_log_file)

    return output


#
# @brief Restarts the line, VT100 terminals only.
#
def reset_line():
    sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


#
# @brief Moves to the next terminal line.
#
def new_line():
    sys.stdout.write('\r\n')
    sys.stdout.flush()


#
# @brief Saves a dictionary to file.
#
# @param[in] d      Dictionary to be written to file.
# @param[in] path   String with the path to the output file.
# @param[in] indent Indent that will be used for the JSON file.
#
# @returns nothing.
def save_dict_as_json(d, path, indent=1, overwrite=False):
    path = os.path.abspath(path)
    if not overwrite:
        assert(not file_exists(path))

    # Convert dict into a JSON string
    contents = json.dumps(d, sort_keys=True, indent=indent)

    # Write output file
    with open(path, "w") as text_file:
        text_file.write(contents)


#
# @brief Loads a JSON of a single section and parameters as a dictionary.
#
# @param[in] path Path to the existing JSON file.
#
# @returns a dictionary with the configuration in the file.
def load_json_as_dict(path):
    path = os.path.abspath(path)
    assert(file_exists(path))

    # Read file
    with open(path, "r") as text_file:
        contents = text_file.read()

    # Convert file into a dict
    d = json.loads(contents)
    assert(isinstance(d, type({})))

    return d


#
# @brief Converts all the keys and values of a dictionary to int.
#
# @param[in] Input dictionary.
#
# @returns a dictionary that contains the same values but of type() int.
def convert_dict_to_int(data):
    return {int(k): int(v) for k, v in data.items()}


#
# @brief Splits a string that is separated by a particular char.
#
# @details Usage:
#          >>> split_at('this_is_my_name_and_its_cool', '_', 2)
#          >>> ('this_is', 'my_name_and_its_cool')
#
#          Code from:
#          https://stackoverflow.com/questions/27227399/python-split-a-string-at-an-underscore
#
# @param[in] s String that you want to split.
# @param[in] c Character that separates the string.
# @param[in] n Number of the split that you want to take.
#
def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


#
# @brief Same effect as the Unix command 'touch'.
#
# @param[in] fname Path to the file.
# @param[in] times (atime, mtime) for the file.
#
# @returns nothing.
def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


#
# @brief Draw a spot on an image.
#
# @param[in] x      Column of the image.
# @param[in] y      Row of the image.
# @param[in] radius Radius of the spot.
# @param[in] colour BGR colour as a list of uint8.
#
# @returns the original image with the sports overlayed on top.
def draw_spot(img, x, y, radius=1, colour=Colour.MAGENTA):
    retval = img.copy()
    cv2.circle(retval, (x, y), radius, colour, thickness=-1)
    return retval


#
# @brief Draw a line on an image.
#
# @param[in] x0        X coordinate of the initial point of the line.
# @param[in] y0        Y coordinate of the initial point of the line.
# @param[in] x1        X coordinate of the final point of the line.
# @param[in] y1        Y coordinate of the final point of the line.
# @param[in] thickness Radius of the spot.
# @param[in] colour    BGR colour as a list of uint8.
#
# @returns the original image with the sports overlayed on top.
def draw_line(img, x0, y0, x1, y1, thickness=1, colour=Colour.MAGENTA):
    retval = img.copy()
    cv2.line(retval, (x0, y0), (x1, y1), colour, thickness)
    return retval


#
# @brief Zips a folder and all its contents to a file.
#
# @param[in] input_dir_path Path to the directory that we will zip.
# @param[in] output_path    Path to the output zip file.
#
# @returns nothing.
def zipdir(input_dir_path, output_path):
    assert(dir_exists(input_dir_path))
    assert(not file_exists(output_path))

    shutil.make_archive(output_path, 'zip', input_dir_path)


#
# @brief Unzip file into a specific path.
#
# @param[in] input_zip_path Path to the zip file to be extracted.
# @param[in] output_dir     Path to the folder where the contents of the zip will be extracted.
#
# @returns nothing.
def unzipdir(input_zip_path, output_dir):
    assert(file_exists(input_zip_path))

    zip_ref = zipfile.ZipFile(input_zip_path)
    zip_ref.extractall(output_dir)
    zip_ref.close()


#
# @brief Generates a random [0, 1].
#
def randbin():
    return np.random.choice([0, 1])


#
# @brief Save string to file.
#
def save_str_to_file(contents, path):
    with open(path, 'w') as text_file:
        text_file.write("{0}".format(contents))


#
# @brief Merge two Python dictionaries.
#
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


#
# @brief Natural sort of a list.
#
# @param[in] l List to sort.
#
# @returns a new list sorted taking into account numbers and not just their ASCII codes.
def natsort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c)
                                for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)


#
# @brief Get the date and time of now.
#
# @param[in] dt_format Format of the date and time.
#
def datetime_str(dt_format='%H_%M_%S__%d_%b_%Y'):
    return datetime.datetime.now().strftime(dt_format)

#
# @brief Convert dictionary into HTML.
#
def convert_dict_to_html(dict_obj, indent=0):
    contents = '  ' * indent + "<ul>\n"
    for k, v in dict_obj.iteritems():
        if isinstance(v, dict):
            contents += ('  ' * indent) + '<li>' + k + ': </li>'
            contents += convert_dict_to_html(v, indent + 1)
        else:
            contents += (
                ' ' * indent) + '<li>' + str(
                    k) + ': ' + str(
                        v) + '</li>'
    contents += '  ' * indent + '</ul>\n'
    return contents


#
# @brief check wether all elements are greater than a number.
#
# @param[in] tensor numpy ndarray.
# @param[in] thresh threshold.
# @param[in] eps    epsilon for comparison.
#
# @returns true if all the values of the tensor are greater than thresh.
#
def tensor_gt(tensor, thresh):
    flat_tensor = tensor.flatten()
    return np.where(flat_tensor > thresh)[0].shape[0] == flat_tensor.shape[0]


#
# @brief check wether all elements are greater or equal than a number.
#
# @param[in] tensor numpy ndarray.
# @param[in] thresh threshold.
# @param[in] eps    epsilon for comparison.
#
# @returns true if all the values of the tensor are greater or equal than thresh.
#
def tensor_gt_eq(tensor, thresh):
    flat_tensor = tensor.flatten()
    return np.where(flat_tensor >= thresh)[0].shape[0] == flat_tensor.shape[0]


#
# @brief Check wether all elements are lower than a number.
#
# @param[in] tensor Numpy ndarray.
# @param[in] thresh Threshold.
# @param[in] eps    Epsilon for comparison.
#
# @returns true if all the values of the tensor are lower than thresh.
#
def tensor_lt(tensor, thresh):
    flat_tensor = tensor.flatten()
    return np.where(flat_tensor < thresh)[0].shape[0] == flat_tensor.shape[0]


#
# @brief Check wether all elements are lower or equal than a number.
#
# @param[in] tensor Numpy ndarray.
# @param[in] thresh Threshold.
# @param[in] eps    Epsilon for comparison.
#
# @returns true if all the values of the tensor are lower or equal than thresh.
#
def tensor_lt_eq(tensor, thresh):
    flat_tensor = tensor.flatten()
    return np.where(flat_tensor <= thresh)[0].shape[0] == flat_tensor.shape[0]


#
# @returns true if a tensor is all zeros. Otherwise false is returned.
#
def tensor_all_zeros(tensor):
    return not tensor.any()


#
# @brief Convert image from PIL to OpenCV.
#
def pil_to_cv2(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGR)


#
# @brief Encode a dictionary into JSON. It must contain either ints, floats, strings, or
#        numpy arrays. Other datatypes will raise a ValueError.
#
def dic_to_json(dic):

    # Build a serialisable dic, converting numpy arrays into
    serialisable_dic = {}
    for k in dic:
        v = dic[k]
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            serialisable_dic[k] = v
        elif isinstance(v, np.ndarray):
            serialisable_dic[k] = v.tolist()
        else:
            raise ValueError(
                'The data type ' + str(type(v)) + ' is not serialisable.')

    return json.dumps(serialisable_dic)


#
# @brief Decode a JSON into a dictionary.
#
def json_to_dic(jsonstr):

    raw_dic = json.loads(jsonstr)
    dic = {}
    for k in raw_dic:
        v = raw_dic[k]
        if isinstance(v, list):
            dic[k] = np.array(v)
        else:
            dic[k] = v

    return dic


#
# @brief Computes the index of the median for a given array.
#
def argmedian(data):
    return np.argsort(data)[len(data) // 2]


def randargmax(b, **kw):
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def randargmin(b, **kw):
    return np.argmin(np.random.random(b.shape) * (b == b.min()), **kw)


#
# @brief Convert a video file to a folder of frames.
#
def convert_video_to_images(
    video_path,
     folder_path,
     fps,
     prefix='',
     fmt='%05d'):
    if dir_exists(folder_path):
        raise RuntimeError(
            'When converting a video to images the output folder should not exist.')
    mkdir(folder_path)
    cmd = 'ffmpeg -i ' + video_path + ' -r ' + str(fps) + ' -f image2 ' + folder_path + '/' \
        + prefix + fmt + '.png'
    return shell(cmd)


#
# @brief Find the nearest element in an array.
#
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


#
# @brief Find the nearest element in an array.
#
def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


# This module cannot be executed as a script because it is not a script :)
if __name__ == "__main__":
    print(
        'Error, this module is not supposed to be executed by itself.',
        sys.stderr)
    sys.exit(1)
