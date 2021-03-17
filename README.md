Introduction
------------
This repository contains code and data related to the paper ["Image Compositing for Segmentation of Surgical Tools without Manual Annotations" by Luis Carlos Garcia Peraza Herrera et al. TMI 2021](https://ieeexplore.ieee.org/document/9350303). Also available in [Arxiv](https://arxiv.org/abs/2102.09528).

Supported platforms
-------------------
This code has been tested with Python 3.8 on Ubuntu 20.04.

Dependencies
------------
* OpenCV with support for Python 3 ([instructions to install it here](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)). Alternatively, you can install it from the Ubuntu repositories with:
```
$ sudo apt install python3-opencv
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
If your foreground images are interlaced you can also use ```--deinterlace 1```, and if they contain noise ```--denoise 1```.

Contact
-------
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
