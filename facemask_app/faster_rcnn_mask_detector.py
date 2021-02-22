import os
import numpy as np
import pandas as pandas
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches

# setting for Starting a Matplotlib GUI outside of the main thread.
# https://ebi-works.com/matplotlib-django/
import matplotlib
matplotlib.use('Agg')


def _get_device():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


class FaceMaskDetector(object):
    def __init__(self, num_classes=3, model_weight_path='../model/model2_cpu.pt'):
        self.num_classes = num_classes
        self.model_weight_path = model_weight_path

    def _get_base_model(self, num_classes):
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True)
        except Exception as exc:
            # TODO add logger
            print(exc)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)
        return model

    def _read_RGB_img(self, input_img_path):
        img = Image.open(input_img_path).convert("RGB")
        return img

    def _prep_img(self, img):
        # TODO resize img
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return data_transforms(img)

    def predict(self, input_img_path):

        model = self._get_base_model(num_classes=self.num_classes)
        model.load_state_dict(torch.load(self.model_weight_path))
        # Inference mode
        model.eval()

        device = _get_device()
        model.to(device)

        img = self._read_RGB_img(input_img_path)
        img_tensor = self._prep_img(img)
        assert img_tensor.ndim == 3

        with torch.no_grad():
            # model input should be list of images.
            preds = model([img_tensor.to(device)])

        return preds

    def save_img_with_anno(self, input_img_path, preds):
        _, ax = plt.subplots(1)

        img = self._read_RGB_img(input_img_path)
        img_tensor = self._prep_img(img)
        img = img_tensor.cpu().data

        # (C, H, W) -> (H, W, C)
        ax.imshow(img.permute(1, 2, 0))

        for box in preds[0]["boxes"]:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.savefig(
            f'../media/output/{os.path.basename(input_img_path)}')

    def detect(self, input_img_path):
        preds = self.predict(input_img_path)
        self.save_img_with_anno(input_img_path, preds)
        # TODO Fix hard coding
        output_file_path = f'../media/output/{os.path.basename(input_img_path)}'
        return output_file_path
