# Face-mask-detection

## Features

### Datasets
- You can download dataset from ["Face Mask Detection" in kaggle datasets](https://www.kaggle.com/andrewmvd/face-mask-detection).

### Training model with jupyter notebook
- You can train Faster RCNN model which can detect face mask in [jupyter notebook](https://github.com/fyk7/face-mask-detection/blob/master/notebook/FasterRCNN_mask_detection.ipynb).

### Face mask detector class
- [Face mask detector](https://github.com/fyk7/face-mask-detection/blob/master/facemask_app/faster_rcnn_mask_detector.py) takes input_path as an argument and returns output_path to the image where the face-mask was detected.

### Flask main file
- Flask app's entry point: [facemask_app/main.py](https://github.com/fyk7/face-mask-detection/blob/master/facemask_app/main.py)

```sh
# You can run face mask detection app with below command.
python3 path/to/facmask_app/main.py
```

### Output example
![](https://github.com/fyk7/face-mask-detection/blob/master/media/output/maksssksksss836.png)
