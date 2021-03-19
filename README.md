Introduction
------------
This repository contains code and data related to the paper ["Image Compositing for Segmentation of Surgical Tools without Manual Annotations" by Luis Carlos Garcia Peraza Herrera et al. TMI 2021](https://ieeexplore.ieee.org/document/9350303). Also available in [Arxiv](https://arxiv.org/abs/2102.09528).

Supported platforms
-------------------
This code has been tested with Python 3.8 on Ubuntu 20.04.

Dependencies
------------
* OpenCV with support for Python 3, which you can install from the Ubuntu repositories with:
```
$ sudo apt install python3-opencv
```
Alternatively, you can install OpenCV from source following:
```bash
# Install OpenCV dependencies
$ sudo apt update
$ sudo apt install build-essential cmake git unzip pkg-config libgtk-3-dev libjpeg-dev libpng-dev libtiff-dev libatlas-base-dev libx264-dev python3-dev libv4l-dev libavcodec-dev libavformat-dev libavresample-dev libswscale-dev libxvidcore-dev gfortran openexr libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

# Download OpenCV
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
$ cd opencv_contrib
$ git checkout 4.5.1
$ cd ../opencv
$ git checkout 4.5.1

# Compile OpenCV
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON ..
$ make -j8

# Install OpenCV
$ sudo make install
$ sudo ldconfig
```

The rest of the dependencies can be installed with pip:
```
$ sudo apt install python3-pip
$ python3 -m pip install synapseclient numpy keras_preprocessing albumentations noise imutils --user
```

Download foreground and background datasets
-------------------------------------------
The datasets are stored in [Synapse](https://synapse.org/synthetic), so you will have to create an account prior to running these scripts:
```
$ python3 src/download_foregrounds.py --username 'your_synapse_username_here' --password 'your_password_here' --output-dir foregrounds
$ python3 src/download_backgrounds.py --username 'your_synapse_username_here' --password 'your_password_here' --output-dir backgrounds
```
Alternatively, foregrounds and backgrounds can be examined and downloaded manually from [here](https://synapse.org/synthetic).

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Generation of semi-synthetic images
-----------------------------------
```
$ python3 src/generate_synthetic_dataset.py --input-fg-dir foregrounds --input-bg-dir backgrounds --output-dir blended --gt-suffix '_seg' --classes 2 --class-map misc/class_map.json --blend-modes '["gaussian", "laplacian", "alpha"]' --num-samples 10 --width 640 --bg-aug '{"horizontal_flip": 1, "vertical_flip": 1, "brightness_range": [0.5, 2.0], "albu_bg": 1.0}' --fg-aug '{"tool_rotation": 180, "tool_zoom": [0.1, 3.5], "tool_shift": 1, "horizontal_flip": 1, "vertical_flip": 1, "brightness_range": [0.5, 2.0], "blood_droplets": 0.5, "albu_fg": 1.0}' --blend-aug '{"blend_border": 0.5, "albu_blend": 0.5}'
```
Change the parameter --num-samples to select the number of semi-synthetic images to blend (100K in the paper).

Chroma key segmentation
-----------------------
If you want to generate your own foreground dataset, you can use the following command to segment your instruments over a chroma key:
```bash
python3 src/chroma.py --input-dir demo_data/foregrounds --output-dir demo_data/foregrounds_segmented --min-hsv-thresh '[35, 70, 15]' --max-hsv-thresh '[95, 255, 255]' --grabcut 1
```
<table align="center">
  <tr>
    <td align="center">Image</td> <td align="center">Output</td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/semi-synthetic/blob/main/demo_data/foregrounds/green.png?raw=true" width=205>
    </td>
    <td align="center">
      <img src="https://github.com/luiscarlosgph/semi-synthetic/blob/main/demo_data/green_seg.png?raw=true" width=205>
    </td>
  </tr>
</table>
This command is based on a simple HSV colour filtering. Run ```$ python3 src/chroma.py --help``` to print the usage information. Typical options are:

1. ```--denoise 1```, blurs the images prior to segmentation. Needed for images captured with real endoscopes.
2. ```--deinterlace 1``` (requires ffmpeg), use it for images captured with old endoscopes.
3. ```--endo-padding 1```, removes the black endoscopic padding. 
4. ```--num-inst 2```, specify the number of instruments (largest connected components kept).  
5. ```--grabcut 1``` GrabCut postprocessing of the masks.

Contact
-------
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
