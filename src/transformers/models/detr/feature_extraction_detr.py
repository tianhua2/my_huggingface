# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature extractor class for DETR."""

import io
import pathlib
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin, is_torch_tensor
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

logger = logging.get_logger(__name__)


ImageInput = Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]]


# 2 functions below inspired by https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def corners_to_center_format(x):
    """
    Converts a NumPy array of bounding boxes of shape (number of bounding boxes, 4) of corners format (x_0, y_0, x_1,
    y_1) to center format (center_x, center_y, width, height).
    """
    x_transposed = x.T
    x0, y0, x1, y1 = x_transposed[0], x_transposed[1], x_transposed[2], x_transposed[3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)


def masks_to_boxes(masks):
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensor, with the boxes in corner (xyxy) format.
    """
    if masks.size == 0:
        return np.zeros((0, 4))

    h, w = masks.shape[-2:]

    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    # see https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    return np.stack([x_min, y_min, x_max, y_max], 1)


# 2 functions below copied from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
# Copyright (c) 2018, Alexander Kirillov
# All rights reserved.
def rgb_to_id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id_to_rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def binary_mask_to_rle(mask):
    """
    Args:
    Converts given binary mask of shape (height, width) to the run-length encoding (RLE) format.
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    if is_torch_tensor(mask):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return [x for x in runs]


def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape (height, width) to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


def check_segment_validity(mask_labels, mask_probs, k, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_area = (mask_probs[k] >= 0.5).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # Weigh each mask by its prediction score
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # Keep track of instances of each class
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # Check if mask exists and large enough to be a segment
        mask_exists, mask_k = check_segment_validity(mask_labels, mask_probs, k, overlap_mask_area_threshold)

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # Add current object segment to final segmentation map
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments


class DetrFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a DETR feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.


    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        max_size (`int`, *optional*, defaults to 1333):
            The largest size an image dimension can have (otherwise it's capped). Only has an effect if `do_resize` is
            set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self,
        format="coco_detection",
        do_resize=True,
        size=800,
        max_size=1333,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.format = self._is_valid_format(format)
        self.do_resize = do_resize
        self.size = size
        self.max_size = max_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]  # ImageNet mean
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]  # ImageNet std

    def _is_valid_format(self, format):
        if format not in ["coco_detection", "coco_panoptic"]:
            raise ValueError(f"Format {format} not supported")
        return format

    def prepare(self, image, target, return_segmentation_masks=False, masks_path=None):
        if self.format == "coco_detection":
            image, target = self.prepare_coco_detection(image, target, return_segmentation_masks)
            return image, target
        elif self.format == "coco_panoptic":
            image, target = self.prepare_coco_panoptic(image, target, masks_path)
            return image, target
        else:
            raise ValueError(f"Format {self.format} not supported")

    # inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L33
    def convert_coco_poly_to_mask(self, segmentations, height, width):

        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            raise ImportError("Pycocotools is not installed in your environment.")

        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = np.asarray(mask, dtype=np.uint8)
            mask = np.any(mask, axis=2)
            masks.append(mask)
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.zeros((0, height, width), dtype=np.uint8)

        return masks

    # inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L50
    def prepare_coco_detection(self, image, target, return_segmentation_masks=False):
        """
        Convert the target in COCO format into the format expected by DETR.
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = np.asarray([image_id], dtype=np.int64)

        # get all COCO annotations for the given image
        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = np.asarray(classes, dtype=np.int64)

        if return_segmentation_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = np.asarray(keypoints, dtype=np.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.reshape((-1, 3))

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if return_segmentation_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["class_labels"] = classes
        if return_segmentation_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = np.asarray([obj["area"] for obj in anno], dtype=np.float32)
        iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno], dtype=np.int64)
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = np.asarray([int(h), int(w)], dtype=np.int64)
        target["size"] = np.asarray([int(h), int(w)], dtype=np.int64)

        return image, target

    def prepare_coco_panoptic(self, image, target, masks_path, return_masks=True):
        w, h = image.size
        ann_info = target.copy()
        ann_path = pathlib.Path(masks_path) / ann_info["file_name"]

        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb_to_id(masks)

            ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
            masks = masks == ids[:, None, None]
            masks = np.asarray(masks, dtype=np.uint8)

            labels = np.asarray([ann["category_id"] for ann in ann_info["segments_info"]], dtype=np.int64)

        target = {}
        target["image_id"] = np.asarray(
            [ann_info["image_id"] if "image_id" in ann_info else ann_info["id"]], dtype=np.int64
        )
        if return_masks:
            target["masks"] = masks
        target["class_labels"] = labels

        target["boxes"] = masks_to_boxes(masks)

        target["size"] = np.asarray([int(h), int(w)], dtype=np.int64)
        target["orig_size"] = np.asarray([int(h), int(w)], dtype=np.int64)
        if "segments_info" in ann_info:
            target["iscrowd"] = np.asarray([ann["iscrowd"] for ann in ann_info["segments_info"]], dtype=np.int64)
            target["area"] = np.asarray([ann["area"] for ann in ann_info["segments_info"]], dtype=np.float32)

        return image, target

    def _resize(self, image, size, target=None, max_size=None):
        """
        Resize the image to the given size. Size can be min_size (scalar) or (w, h) tuple. If size is an int, smaller
        edge of the image will be matched to this number.

        If given, also resize the target accordingly.
        """
        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size
            else:
                # size returned must be (w, h) since we use PIL to resize images
                # so we revert the tuple
                return get_size_with_aspect_ratio(image_size, size, max_size)[::-1]

        size = get_size(image.size, size, max_size)
        rescaled_image = self.resize(image, size=size)

        if target is None:
            return rescaled_image, None

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        w, h = size
        target["size"] = np.asarray([h, w], dtype=np.int64)

        if "masks" in target:
            # use PyTorch as current workaround
            # TODO replace by self.resize
            masks = torch.from_numpy(target["masks"][:, None]).float()
            interpolated_masks = nn.functional.interpolate(masks, size=(h, w), mode="nearest")[:, 0] > 0.5
            target["masks"] = interpolated_masks.numpy()

        return rescaled_image, target

    def _normalize(self, image, mean, std, target=None):
        """
        Normalize the image with a certain mean and std.

        If given, also normalize the target bounding boxes based on the size of the image.
        """

        image = self.normalize(image, mean=mean, std=std)
        if target is None:
            return image, None

        target = target.copy()
        h, w = image.shape[-2:]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = corners_to_center_format(boxes)
            boxes = boxes / np.asarray([w, h, w, h], dtype=np.float32)
            target["boxes"] = boxes

        return image, target

    def __call__(
        self,
        images: ImageInput,
        annotations: Union[List[Dict], List[List[Dict]]] = None,
        return_segmentation_masks: Optional[bool] = False,
        masks_path: Optional[pathlib.Path] = None,
        pad_and_return_pixel_mask: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s) and optional annotations. Images are by default
        padded up to the largest image in a batch, and a pixel mask is created that indicates which pixels are
        real/which are padding.

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            annotations (`Dict`, `List[Dict]`, *optional*):
                The corresponding annotations in COCO format.

                In case [`DetrFeatureExtractor`] was initialized with `format = "coco_detection"`, the annotations for
                each image should have the following format: {'image_id': int, 'annotations': [annotation]}, with the
                annotations being a list of COCO object annotations.

                In case [`DetrFeatureExtractor`] was initialized with `format = "coco_panoptic"`, the annotations for
                each image should have the following format: {'image_id': int, 'file_name': str, 'segments_info':
                [segment_info]} with segments_info being a list of COCO panoptic annotations.

            return_segmentation_masks (`Dict`, `List[Dict]`, *optional*, defaults to `False`):
                Whether to also include instance segmentation masks as part of the labels in case `format =
                "coco_detection"`.

            masks_path (`pathlib.Path`, *optional*):
                Path to the directory containing the PNG files that store the class-agnostic image segmentations. Only
                relevant in case [`DetrFeatureExtractor`] was initialized with `format = "coco_panoptic"`.

            pad_and_return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).
            - **labels** -- Optional labels to be fed to a model (when `annotations` are provided)
        """
        # Input type checking for clearer error

        valid_images = False
        valid_annotations = False
        valid_masks_path = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        # Check that annotations has a valid type
        if annotations is not None:
            if not is_batched:
                if self.format == "coco_detection":
                    if isinstance(annotations, dict) and "image_id" in annotations and "annotations" in annotations:
                        if isinstance(annotations["annotations"], (list, tuple)):
                            # an image can have no annotations
                            if len(annotations["annotations"]) == 0 or isinstance(annotations["annotations"][0], dict):
                                valid_annotations = True
                elif self.format == "coco_panoptic":
                    if isinstance(annotations, dict) and "image_id" in annotations and "segments_info" in annotations:
                        if isinstance(annotations["segments_info"], (list, tuple)):
                            # an image can have no segments (?)
                            if len(annotations["segments_info"]) == 0 or isinstance(
                                annotations["segments_info"][0], dict
                            ):
                                valid_annotations = True
            else:
                if isinstance(annotations, (list, tuple)):
                    if len(images) != len(annotations):
                        raise ValueError("There must be as many annotations as there are images")
                    if isinstance(annotations[0], Dict):
                        if self.format == "coco_detection":
                            if isinstance(annotations[0]["annotations"], (list, tuple)):
                                valid_annotations = True
                        elif self.format == "coco_panoptic":
                            if isinstance(annotations[0]["segments_info"], (list, tuple)):
                                valid_annotations = True

            if not valid_annotations:
                raise ValueError(
                    """
                    Annotations must of type `Dict` (single image) or `List[Dict]` (batch of images). In case of object
                    detection, each dictionary should contain the keys 'image_id' and 'annotations', with the latter
                    being a list of annotations in COCO format. In case of panoptic segmentation, each dictionary
                    should contain the keys 'file_name', 'image_id' and 'segments_info', with the latter being a list
                    of annotations in COCO format.
                    """
                )

        # Check that masks_path has a valid type
        if masks_path is not None:
            if self.format == "coco_panoptic":
                if isinstance(masks_path, pathlib.Path):
                    valid_masks_path = True
                if not valid_masks_path:
                    raise ValueError(
                        "The path to the directory containing the mask PNG files should be provided as a"
                        " `pathlib.Path` object."
                    )

        if not is_batched:
            images = [images]
            if annotations is not None:
                annotations = [annotations]

        # Create a copy of the list to avoid editing it in place
        images = [image for image in images]

        if annotations is not None:
            annotations = [annotation for annotation in annotations]

        # prepare (COCO annotations as a list of Dict -> DETR target as a single Dict per image)
        if annotations is not None:
            for idx, (image, target) in enumerate(zip(images, annotations)):
                if not isinstance(image, Image.Image):
                    image = self.to_pil_image(image)
                image, target = self.prepare(image, target, return_segmentation_masks, masks_path)
                images[idx] = image
                annotations[idx] = target

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            if annotations is not None:
                for idx, (image, target) in enumerate(zip(images, annotations)):
                    image, target = self._resize(image=image, target=target, size=self.size, max_size=self.max_size)
                    images[idx] = image
                    annotations[idx] = target
            else:
                for idx, image in enumerate(images):
                    images[idx] = self._resize(image=image, target=None, size=self.size, max_size=self.max_size)[0]

        if self.do_normalize:
            if annotations is not None:
                for idx, (image, target) in enumerate(zip(images, annotations)):
                    image, target = self._normalize(
                        image=image, mean=self.image_mean, std=self.image_std, target=target
                    )
                    images[idx] = image
                    annotations[idx] = target
            else:
                images = [
                    self._normalize(image=image, mean=self.image_mean, std=self.image_std)[0] for image in images
                ]
        else:
            images = [np.array(image) for image in images]

        if pad_and_return_pixel_mask:
            # pad images up to largest image in batch and create pixel_mask
            max_size = self._max_by_axis([list(image.shape) for image in images])
            c, h, w = max_size
            padded_images = []
            pixel_mask = []
            for image in images:
                # create padded image
                padded_image = np.zeros((c, h, w), dtype=np.float32)
                padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
                padded_images.append(padded_image)
                # create pixel mask
                mask = np.zeros((h, w), dtype=np.int64)
                mask[: image.shape[1], : image.shape[2]] = True
                pixel_mask.append(mask)
            images = padded_images

        # return as BatchFeature
        data = {}
        data["pixel_values"] = images
        if pad_and_return_pixel_mask:
            data["pixel_mask"] = pixel_mask
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        if annotations is not None:
            # Convert to TensorType
            tensor_type = return_tensors
            if not isinstance(tensor_type, TensorType):
                tensor_type = TensorType(tensor_type)

            if not tensor_type == TensorType.PYTORCH:
                raise ValueError("Only PyTorch is supported for the moment.")
            else:
                if not is_torch_available():
                    raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")

                encoded_inputs["labels"] = [
                    {k: torch.from_numpy(v) for k, v in target.items()} for target in annotations
                ]

        return encoded_inputs

    def _max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def pad_and_create_pixel_mask(
        self, pixel_values_list: List["torch.Tensor"], return_tensors: Optional[Union[str, TensorType]] = None
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        Args:
            pixel_values_list (`List[torch.Tensor]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape (C, H, W).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).

        """

        max_size = self._max_by_axis([list(image.shape) for image in pixel_values_list])
        c, h, w = max_size
        padded_images = []
        pixel_mask = []
        for image in pixel_values_list:
            # create padded image
            padded_image = np.zeros((c, h, w), dtype=np.float32)
            padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
            padded_images.append(padded_image)
            # create pixel mask
            mask = np.zeros((h, w), dtype=np.int64)
            mask[: image.shape[1], : image.shape[2]] = True
            pixel_mask.append(mask)

        # return as BatchFeature
        data = {"pixel_values": padded_images, "pixel_mask": pixel_mask}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    # POSTPROCESSING METHODS
    # inspired by https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258
    def post_process(self, outputs, target_sizes):
        """
        Converts the output of [`DetrForObjectDetection`] into the format expected by the COCO api. Only supports
        PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        warnings.warn(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection`",
            FutureWarning,
        )

        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        return results

    def post_process_segmentation(self, outputs, target_sizes, threshold=0.9, mask_threshold=0.5):
        """
        Converts the output of [`DetrForSegmentation`] into image segmentation predictions. Only supports PyTorch.

        Parameters:
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):
                Torch Tensor (or list) corresponding to the requested final size (h, w) of each prediction.
            threshold (`float`, *optional*, defaults to 0.9):
                Threshold to use to filter out queries.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, and masks for an image
            in the batch as predicted by the model.
        """
        warnings.warn(
            "`post_process_segmentation` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_semantic_segmentation`.",
            FutureWarning,
        )
        out_logits, raw_masks = outputs.logits, outputs.pred_masks
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, size in zip(out_logits, raw_masks, target_sizes):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs.logits.shape[-1] - 1) & (scores > threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_masks = (cur_masks.sigmoid() > mask_threshold) * 1

            predictions = {"scores": cur_scores, "labels": cur_classes, "masks": cur_masks}
            preds.append(predictions)
        return preds

    # inspired by https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L218
    def post_process_instance(self, results, outputs, orig_target_sizes, max_target_sizes, threshold=0.5):
        """
        Converts the output of [`DetrForSegmentation`] into actual instance segmentation predictions. Only supports
        PyTorch.

        Args:
            results (`List[Dict]`):
                Results list obtained by [`~DetrFeatureExtractor.post_process`], to which "masks" results will be
                added.
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            orig_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation).
            max_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the maximum size (h, w) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation).
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, boxes and masks for an
            image in the batch as predicted by the model.
        """
        warnings.warn(
            "`post_process_instance` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_instance_segmentation`.",
            FutureWarning,
        )

        if len(orig_target_sizes) != len(max_target_sizes):
            raise ValueError("Make sure to pass in as many orig_target_sizes as max_target_sizes")
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs.pred_masks.squeeze(2)
        outputs_masks = nn.functional.interpolate(
            outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        outputs_masks = (outputs_masks.sigmoid() > threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = nn.functional.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results

    # inspired by https://github.com/facebookresearch/detr/blob/master/models/segmentation.py#L241
    def post_process_panoptic(self, outputs, processed_sizes, target_sizes=None, is_thing_map=None, threshold=0.85):
        """
        Converts the output of [`DetrForSegmentation`] into actual panoptic predictions. Only supports PyTorch.

        Parameters:
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            processed_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):
                Torch Tensor (or list) containing the size (h, w) of each image of the batch, i.e. the size after data
                augmentation but before batching.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`, *optional*):
                Torch Tensor (or list) corresponding to the requested final size (h, w) of each prediction. If left to
                None, it will default to the `processed_sizes`.
            is_thing_map (`torch.Tensor` of shape `(batch_size, 2)`, *optional*):
                Dictionary mapping class indices to either True or False, depending on whether or not they are a thing.
                If not set, defaults to the `is_thing_map` of COCO panoptic.
            threshold (`float`, *optional*, defaults to 0.85):
                Threshold to use to filter out queries.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing a PNG string and segments_info values for
            an image in the batch as predicted by the model.
        """
        warnings.warn(
            "`post_process_panoptic is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_panoptic_segmentation`.",
            FutureWarning,
        )
        if target_sizes is None:
            target_sizes = processed_sizes
        if len(processed_sizes) != len(target_sizes):
            raise ValueError("Make sure to pass in as many processed_sizes as target_sizes")

        if is_thing_map is None:
            # default to is_thing_map of COCO panoptic
            is_thing_map = {i: i <= 90 for i in range(201)}

        out_logits, raw_masks, raw_boxes = outputs.logits, outputs.pred_masks, outputs.pred_boxes
        if not len(out_logits) == len(raw_masks) == len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits and masks"
            )
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs.logits.shape[-1] - 1) & (scores > threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = center_to_corners_format(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            if len(cur_boxes) != len(cur_classes):
                raise ValueError("Not as many boxes as there are classes")

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id_to_rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes()))
                np_seg_img = np_seg_img.view(final_h, final_w, 3)
                np_seg_img = np_seg_img.numpy()

                m_id = torch.from_numpy(rgb_to_id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds

    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None
    ):
        """
        Converts the output of [`DetrForObjectDetection`] into the format expected by the COCO api. Only supports
        PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*, defaults to `None`):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple[int, int]] = None):
        """
        Args:
        Converts the output of [`DetrForSegmentation`] into semantic segmentation maps. Only supports PyTorch.
            outputs ([`DetrForSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*, defaults to `None`):
                A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the
                batch. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Args:
        Converts the output of [`DetrForSegmentation`] into instance segmentation predictions. Only supports PyTorch.
            outputs ([`DetrForSegmentation`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_coco_annotation (`bool`, *optional*):
                Defaults to `False`. If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE)
                format.
        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id` or
              `List[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
              `True`. Set to `None` if no mask if found above `threshold`.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- An integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_probs_item,
                pred_scores_item,
                pred_labels_item,
                overlap_mask_area_threshold,
                target_size,
            )

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict]:
        """
        Args:
        Converts the output of [`DetrForSegmentation`] into image panoptic segmentation predictions. Only supports
        PyTorch.
            outputs ([`DetrForSegmentation`]):
                The outputs from [`DetrForSegmentation`].
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.
        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id` or
              `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized to
              the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """

        if label_ids_to_fuse is None:
            warnings.warn("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_probs_item,
                pred_scores_item,
                pred_labels_item,
                overlap_mask_area_threshold,
                label_ids_to_fuse,
                target_size,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results
