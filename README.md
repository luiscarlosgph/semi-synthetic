# Introduction
This repository contains code and data related to the paper "Image Compositing for Segmentation of Surgical Tools without Manual Annotations" by Luis Carlos Garcia Peraza Herrera et al.

# Generation of semi-synthetic images
```
$ python src/generate_synthetic_dataset.py --input-fg-dir foregrounds --input-bg-dir backgrounds --output-dir blended --gt-suffix '_seg' --classes 2 --class-map misc/class_map.json --blend-modes '["gaussian", "laplacian", "alpha"]' --num-samples 10 --width 640 --bg-aug '{"horizontal_flip": 1, "vertical_flip": 1, "brightness_range": [0.5, 2.0], "albu_bg": 1.0}' --fg-aug '{"tool_rotation": 180, "tool_zoom": [0.1, 3.5], "tool_shift": 1, "horizontal_flip": 1, "vertical_flip": 1, "brightness_range": [0.5, 2.0], "blood_droplets": 0.5, "albu_fg": 1.0}' --blend-aug '{"blend_border": 0.5, "albu_blend": 0.5}'
```

# Dataset
[Click here to access the dataset](https://synapse.org/synthetic)

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

# Contact
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
