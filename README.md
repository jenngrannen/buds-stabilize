# Keypoint Regression for Bimanual Stabilization

This repo provides a lightweight, general framework for supervised keypoint regression, specifically for predicting a stabilizing position for bimanual robotic manipulation.

### Getting Started/Overview
#### Dataset Generation
* This is best done locally, instead of on a remote host since we use a OpenCV mouse GUI to annotate images and get paired (image, keypoint) datasets. 
* Use the script `python annotate_real.py` which expects a folder called `images` (move `train/images` or `test/images` to the same directory level as this script). It will launch an OpenCV window where you can annotate keypoints; double click to annotate/save a point, which will be visualized as a blue circle on the image. Note that the script is currently configured to handle one single keypoint, and will automatically go to the next image once a click is recorded. Press `s` to skip an image and `r` to clear all annotations. The script saves the images/annotations to a folder called `real_data` organized as follows:
```
real_data/
|-- images
|   `-- 00000.jpg
|   ...
`-- keypoints
    `-- 00000.npy
    ...
```
* Use the script `augment.py` which expects a folder called `images` and `keypoints` (copy `real_data/images` and `real_data/keypoints` to the same directory level as this script). It will use image space and affine transformations to augment the dataset by `num_augs_per_img` and directly output the augmented images and keypoint annotations to the same `images` and `keypoints` folders
* Finally, move `images` and `keypoints` to the folder `train`
* Repeat the above steps on the test split
* Move the folders  `test`  and `train` to a folder with your desired dataset name
* This should produce a dataset like so:
```
<your_dataset_name>
|-- test
|   |-- images
|   `-- keypoints
`-- train
    |-- images
    `-- keypoints
```

#### Training and Inference
* Set up an environment according to the dependencies listed in `requirements.txt`
* Configure `train.py` by replacing `dataset_dir = <your_dataset_name>`
* Run `python train.py`
* This will save checkpoints to `checkpoints/<your_dataset_name>`
* Update `analysis.py` by with `keypoints.load_state_dict(torch.load('checkpoints/<your_dataset_name>/model_2_1_24_<final_test_loss>.pth'))`
* Run `python analysis.py` which will save predicted heatmap keypoint predictions to `preds`
