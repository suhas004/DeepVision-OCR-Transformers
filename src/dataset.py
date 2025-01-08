# Importing required libraries
from __future__ import division, print_function
import os
import sys
import json
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import albumentations as A
from transformers import TrOCRProcessor

# Adding path to system path
sys.path.append("/mnt/suhas/OCR/github/trocr_train/src")

# Importing local modules
from configs import paths, constants
from util import debug_print
#garbage collection
import gc

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms


class Erosion(A.ImageOnlyTransform):
    """
    Apply erosion operation to an image.

    Erosion is a morphological operation that shrinks the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the erosion kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the erosion kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(A.ImageOnlyTransform):
    """
    Apply dilation operation to an image.

    Dilation is a morphological operation that expands the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the dilation kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the dilation kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(A.ImageOnlyTransform):
    """
    Apply a bitmap-style transformation to an image.

    This transformation replaces all pixel values below a certain threshold with a specified value.

    Args:
        value (int, optional): The value to replace pixels below the threshold with. Default is 0.
        lower (int, optional): The threshold value below which pixels will be replaced. Default is 200.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img


import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
def resize_and_pad_image(img, target_size=(128, 128), content_size=(128, 32)):
    img = Image.fromarray(img).convert('L')  # Convert to grayscale (mode 'L')

    resized_content = img.resize((content_size[0], content_size[1]), Image.ANTIALIAS)

    canvas = Image.new('L', target_size, 0)  # Create a black canvas (grayscale mode)
    x_offset = (target_size[0] - content_size[0]) // 2
    y_offset = (target_size[1] - content_size[1]) // 2
    canvas.paste(resized_content, (x_offset, y_offset))
    return canvas



def train_transform(img):
    transform = A.Compose([
        A.RandomScale(scale_limit=(-0.1, 0.2), interpolation=1, p=0.1),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.2),
        A.ColorJitter(brightness=0.0, contrast=0.2, saturation=0.2, hue=0.1, p=0.08),
        A.Perspective(scale=(0.01, 0.02), p=0.1),
        A.OneOf([A.GaussNoise(p=0.5), A.Blur(p=0.5, blur_limit=1)], p=0.07),
        A.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.1),
        A.ImageCompression(95, p=0.07),
        A.Compose([
            A.Affine(translate_px=(0, 5), always_apply=True, cval=(255, 255, 255)),
            A.ElasticTransform(p=1, alpha=50, sigma=120 * 0.1, alpha_affine=120 * 0.01,
                               border_mode=0, value=(255, 255, 255)),
        ], p=0.08),
        A.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.04),
        A.ShiftScaleRotate(shift_limit_x=(0, 0.04), shift_limit_y=(0, 0.03),
                           scale_limit=(-0.15, 0.03), rotate_limit=2, border_mode=0,
                           interpolation=2, value=(255, 255, 255), p=0.00),
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # RGB mean and std
        # ToTensorV2(),
    ], p=0.7)
    
    #read the image suing img_name and convert to numpy array
    img = Image.open(img).convert('L')
    img_np = np.array(img)

    # img_np = np.array(img)
    augumented_image = transform(image=img_np)["image"]

    img = resize_and_pad_image(augumented_image).convert('L')

    img = np.array(img)


    mean = np.array([0.5])
    std = np.array([0.5])
    img_tensor = (img - mean) / std
    img_tensor = img_tensor / 255.0

    # img_tensor = img_tensor / 255.0

    # img_tensor = ToTensorV2()(image=img)["image"]
    # img_tensor = torch.from_numpy(img)
    img_tensor = transforms.ToTensor()(img_tensor)

    
    #Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) image_tensor

    del img_np, img
    return img_tensor

def test_transform(img, img_name="None"):

    img = Image.open(img).convert('RGB')
    img_np = np.array(img)

    img = resize_and_pad_image(img_np)
    img = np.array(img)

   
    mean = np.array([0.5])
    std = np.array([0.5])
    img_tensor = (img - mean) / std
    img_tensor = img_tensor / 255.0

    img_tensor = transforms.ToTensor()(img_tensor)


    del img_np, img
    return img_tensor





class HCRDataset(Dataset):
    def __init__(self, data_dir: Path, processor: TrOCRProcessor):
        # data_dir = '/mnt/suhas/OCR/trocr/dataset/train/train_combined.json'

        self.json = json.load(open(data_dir))
        self.processor = processor
        self.label_list = list(self.json.values())
        self._max_label_len = max(
            [constants.word_len_padding] + [len(label) for label in self.label_list]
        )
        self.image_paths = list(self.json.keys())
        self.augmentations = True

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label = self.json[img_name]
        try:
            # image = Image.open(img_name).convert("RGB")
            image = Image.open(img_name).convert("RGB")
            if self.augmentations:
                image_tensor = train_transform(img_name)
            else:
                image_tensor = test_transform(img_name)

            if image_tensor is None:
                image.save("/mnt/suhas/OCR/trocr_train/src/fail/"+img_name.split("/")[-1])
                return None

            # augmented_image = augmented_image.convert("RGB")

            # image_tensor: torch.Tensor = self.processor(
            #     augmented_image, return_tensors="pt"
            # ).pixel_values[0]
        except Exception as e:
            # save the image in a folder
            print(e)
            # print(image_tensor.size()  )
            image.save("/mnt/suhas/OCR/trocr_train/src/fail/" + img_name.split("/")[-1])
            return None

        label_tensor = self.processor.tokenizer(
            label,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self._max_label_len,
        ).input_ids[0]

        del image

        return {"idx": idx, "input": image_tensor, "label": label_tensor}

    def get_label(self, idx) -> str:
        assert (
            0 <= idx < len(self.label_list)
        ), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.label_list[idx]

    def get_path(self, idx) -> str:
        assert (
            0 <= idx < len(self.label_list)
        ), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.image_paths[idx]

    def augment_image(self, img, img_name):
        transform = A.Compose(
            [
                A.RandomScale(scale_limit=(-0.1, 0.2), interpolation=1, p=0.7),
                A.CLAHE(
                    clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.6
                ),
                A.RandomShadow(
                    p=0.5, num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=2
                ),
                A.ColorJitter(
                    brightness=0.0, contrast=0.2, saturation=0.2, hue=0.1, p=0.4
                ),
                A.Perspective(scale=(0.01, 0.02), p=0.5),
                A.OneOf([A.GaussNoise(p=0.5), A.Blur(p=0.5, blur_limit=1)]),
            ],
            p=0.8,
        )
        img_np = np.array(img)  # Convert PIL Image to NumPy array
        augmented = transform(image=img_np)["image"]

        del img_np

        return Image.fromarray(augmented).convert("RGB")


class HCRDataset_test(Dataset):
    def __init__(self, data_dir: Path, processor: TrOCRProcessor):
        # data_dir = '/mnt/suhas/OCR/trocr/dataset/trail.json'
        self.json = json.load(open(data_dir))
        self.processor = processor
        self.label_list = list(self.json.values())
        self._max_label_len = max(
            [constants.word_len_padding] + [len(label) for label in self.label_list]
        )
        self.image_paths = list(self.json.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label = self.json[img_name]

        # image = Image.open(img_name).convert("RGB")
        # image_tensor: torch.Tensor = self.processor(
        #     image, return_tensors="pt"
        # ).pixel_values[0]
        # image = resize_and_pad_image(img_name)
 
        image_tensor = test_transform(img_name)


        label_tensor = self.processor.tokenizer(
            label,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self._max_label_len,
        ).input_ids[0]

        # del image

        return {"idx": idx, "input": image_tensor, "label": label_tensor}

    def get_label(self, idx) -> str:
        assert (
            0 <= idx < len(self.label_list)
        ), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.label_list[idx]

    def get_path(self, idx) -> str:
        assert (
            0 <= idx < len(self.label_list)
        ), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.image_paths[idx]
