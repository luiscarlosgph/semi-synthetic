# Introduction
This repository contains code and data related to the paper "Image Compositing for Segmentation of Surgical Tools without Manual Annotations" by Luis Carlos Garcia Peraza Herrera et al.

# Supported platforms
This code has been tested in Ubuntu 16.04 and Ubuntu 18.04.

# Install dependencies
```
$ pip3 install synapseclient opencv-python numpy keras_preprocessing albumentations noise imutils --user
```

# Download foreground and background datasets
The datasets are stored in [Synapse](https://synapse.org/synthetic), so you will have to create an account prior to running these scripts:
```
python3 src/download_foregrounds.py --username 'your_synapse_username_here' --password 'your_password_here' --output-dir foregrounds
python3 src/download_backgrounds.py --username 'your_synapse_username_here' --password 'your_password_here' --output-dir backgrounds
```
Alternatively, foregrounds and backgrounds can be examined and downloaded manually from [here](https://synapse.org/synthetic).

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

# Generation of semi-synthetic images
```
$ python3 src/generate_synthetic_dataset.py --input-fg-dir foregrounds --input-bg-dir backgrounds --output-dir blended --gt-suffix '_seg' --classes 2 --class-map misc/class_map.json --blend-modes '["gaussian", "laplacian", "alpha"]' --num-samples 10 --width 640 --bg-aug '{"horizontal_flip": 1, "vertical_flip": 1, "brightness_range": [0.5, 2.0], "albu_bg": 1.0}' --fg-aug '{"tool_rotation": 180, "tool_zoom": [0.1, 3.5], "tool_shift": 1, "horizontal_flip": 1, "vertical_flip": 1, "brightness_range": [0.5, 2.0], "blood_droplets": 0.5, "albu_fg": 1.0}' --blend-aug '{"blend_border": 0.5, "albu_blend": 0.5}'
```
Change the parameter --num-samples to select the number of semi-synthetic images to blend (100000 in the paper).


# Contact
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
