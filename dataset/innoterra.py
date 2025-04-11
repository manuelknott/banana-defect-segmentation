import os
from copy import deepcopy
import torchvision.ops
import yaml
import json
from typing import Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from torchvision import tv_tensors

from scipy.ndimage import binary_dilation
from pycocotools import mask as mask_utils

from utils.custom_coco import COCO
from utils.visualizer import SegmentationMapVisualizer
from utils.crop_augmentation import get_padding_mask

config = yaml.safe_load(open("config.yaml"))


class InnoterraDataset(Dataset):
    """
    Image (semantic) segmentation dataset.
    """

    root_dir = config["datasets"]["innoterra"]["root_dir"]
    img_dir = os.path.join(root_dir, "resized_images_1024")

    defect_indexes = [3, 4, 5, 6, 7]
    ignore_index = 255

    def __init__(self, sample_ids=None, augment=True, color_augment=True, resolution=1024,
                 defect_ignore_pad_px=0, annotation_type="masks",
                 separate_background_banana=True,
                 separate_defect_types=False,
                 merge_defect_masks_into_bananas=False,
                 no_transform=False,
                 defect_mask_source="annotated",
                 preprocess_defects=False,  # recommended when only using a single defect class
                 ):

        assert defect_mask_source in ["annotated", "sam-b", "sam-l", "sam-h", "sam2-l"]
        if defect_mask_source == "annotated":
            self.annotations_path = os.path.join(self.root_dir, "annotations.json")
        elif defect_mask_source == "sam-b":
            self.annotations_path = os.path.join(self.root_dir, "annotations_sam_vit_b.json")
        elif defect_mask_source == "sam-l":
            self.annotations_path = os.path.join(self.root_dir, "annotations_sam_vit_l.json")
        elif defect_mask_source == "sam-h":
            self.annotations_path = os.path.join(self.root_dir, "annotations_sam_vit_h.json")
        elif defect_mask_source == "sam2-l":
            self.annotations_path = os.path.join(self.root_dir, "annotations_sam2_vit_l.json")
        else:
            raise ValueError(f"Unknown defect_mask_source: {defect_mask_source}")

        if annotation_type == "panoptic" and defect_ignore_pad_px > 0:
            raise NotImplementedError("Panoptic segmentation does not support defect_ignore_pad_px")

        self.augment = augment
        self.resolution = resolution

        img_order_filepath = 'dataset/image_order.txt'
        if os.path.isfile(img_order_filepath):
            with open(img_order_filepath, 'r') as f:
                self.image_file_names = f.read().splitlines()
        else:
            # careful, os.listdir yields different orders depending on OS
            self.image_file_names = [f for f in os.listdir(self.img_dir)]

        self.image_file_names = [f for f in self.image_file_names if f.endswith('.jpg')]

        self.defect_ignore_pad_px = defect_ignore_pad_px

        self.separate_background_banana = separate_background_banana
        self.separate_defect_types = separate_defect_types
        self.merge_defect_masks_into_bananas = merge_defect_masks_into_bananas
        self.no_transform = no_transform
        self.preprocess_defects = preprocess_defects

        assert annotation_type in ["masks", "bboxes", "masks+bboxes", "panoptic", "bboxes-detr"], \
            "annotation_type must be 'masks' or 'bboxes' or 'masks+bboxes', or 'panoptic' or 'bboxes-detr'"
        self.annotation_type = annotation_type
        self.annotations = json.load(open(self.annotations_path))

        self.filenames_to_ids = {i['file_name']: i['id'] for i in self.annotations['images']}
        self.ids_to_shapes = {i['id']: (i['height'], i['width']) for i in self.annotations['images']}

        if sample_ids is not None:  # used to sample train_test_split
            self.image_file_names = [self.image_file_names[i] for i in sample_ids]
        #            print(self.image_file_names)
        #            exit()

        if augment:
            self.transform = T2.Compose([
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True),
                T2.RandomCrop((1024, 1024), pad_if_needed=True),
                T2.Resize(self.resolution, antialias=True),
                T2.RandomHorizontalFlip(0.5)
            ])
        else:
            self.transform = T2.Compose([
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True),
                T2.CenterCrop((1024, 1024)),
                T2.Resize(self.resolution, antialias=True),
            ])

        if color_augment:
            CAUG_FACTOR = 0.5
            self.color_transform = T.Compose([
                T.ColorJitter(brightness=0.2 * CAUG_FACTOR,
                              contrast=0.2 * CAUG_FACTOR,
                              saturation=0.2 * CAUG_FACTOR,
                              hue=0.1 * CAUG_FACTOR),
                T.RandomAdjustSharpness(2 * CAUG_FACTOR),
                # T.RandomEqualize(),
                # T.RandomAutocontrast()
            ])
        else:
            self.color_transform = None

        class_names = []
        if not self.annotation_type == "bboxes":
            class_names.append("background")
            if self.separate_background_banana:
                class_names.append("background_banana")
            class_names.append("foreground_banana")
        if self.separate_defect_types:
            class_names.extend(["old bruise", "old scar", "new bruise", "new scar"])
        else:
            class_names.append("defect")
        self.class_names = class_names

    def _prepare_coco_annotations(self):
        annotations = deepcopy(self.annotations)
        annotations["images"] = [i for i in annotations["images"] if i['file_name'] in self.image_file_names]
        valid_image_ids = {i['id'] for i in annotations["images"]}

        if self.separate_defect_types:
            defect_names = ["old bruise", "old scar", "new bruise", "new scar"]
        else:
            defect_names = ["defect"]
        annotations["categories"] = [{"id": i, "name": name} for i, name in enumerate(defect_names)]

        new_annotations = list()
        for a in annotations["annotations"]:
            if a['image_id'] not in valid_image_ids or a['category_id'] not in [3, 4, 5,
                                                                                6]:  # is there still a unclassified defect?
                continue
            a['category_id'] = a['category_id'] - 3 if self.separate_defect_types else 0
            a["area"] = mask_utils.decode(a["segmentation"]).sum()  # TODO hotfix
            new_annotations.append(a)

        annotations["annotations"] = new_annotations
        return annotations

    @property
    def banana_ids(self):
        return {1, 2} if self.separate_background_banana else {1}

    @property
    def defect_ids(self):
        if self.separate_defect_types and self.separate_background_banana:
            return {3, 4, 5, 6}
        elif self.separate_defect_types and not self.separate_background_banana:
            return {2, 3, 4, 5}
        elif not self.separate_defect_types and self.separate_background_banana:
            return {3}
        else:
            return {2}

    @property
    def class_dict(self):
        return {i: self.class_names[i] for i in range(len(self.class_names))}

    @staticmethod
    def _is_foreground_defect(defect_mask: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray, iters=1) -> bool:
        """
        Applies binary dilation to the defect mask and checks if it intersects with the foreground mask
        To address edge cases, we decide that a defect is foreground if the FG dilation overlap is bigger than BG dilation overlap.

        All arrays are expected to be binary (True, False)
        """
        dilated_mask = binary_dilation(defect_mask, iterations=iters)
        fg_overlap = np.sum(dilated_mask * fg_mask)
        bg_overlap = np.sum(dilated_mask * bg_mask)
        return fg_overlap > bg_overlap

    @staticmethod
    def _separate_masks(global_mask: np.ndarray, dilation_iters=4) -> np.ndarray:
        if dilation_iters == 0:
            defects_mask_uint = np.where(global_mask, 255, 0).astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(defects_mask_uint)
            return labels_im

        defects_mask_dilated = binary_dilation(global_mask, iterations=dilation_iters)
        defects_mask_uint = np.where(defects_mask_dilated, 255, 0).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(defects_mask_uint)

        return np.where(global_mask, labels_im, 0)

    def _merge_classes(self, x: np.ndarray) -> np.ndarray:

        if self.annotation_type == "bboxes" or self.annotation_type == "bboxes-detr":
            if not self.separate_defect_types:
                x[x > 2] = 3
            x -= 3  # Move the defects class to a new label starting from 0

        if self.merge_defect_masks_into_bananas:
            if not self.separate_background_banana:
                x[(x > 1) & (x < 255)] = 1
                return x
            else:
                out = x.copy()
                bg_mask = (x == 1)
                fg_mask = (x == 2)
                for defect_id in [3, 4, 5, 6, 7]:
                    if not (x == defect_id).any():
                        continue
                    semantic_defect_mask = (x == defect_id)
                    instance_defect_mask = self._separate_masks(semantic_defect_mask)
                    for instance_id in np.unique(instance_defect_mask):
                        if instance_id == 0:
                            continue
                        if self._is_foreground_defect(instance_defect_mask == instance_id, fg_mask, bg_mask):
                            out[instance_defect_mask == instance_id] = 2
                        else:
                            out[instance_defect_mask == instance_id] = 1
                return out

        if not self.separate_defect_types:
            x[(x >= 3) & (x < 255)] = 3

        if not self.separate_background_banana:
            # Merging background banana and foreground banana into single class
            x[x == 2] = 1
            x[(x > 1) & (x < 255)] -= 1  # Move the defects class to a new label

        return x

    @classmethod
    def n_samples_total(cls):
        return len([f for f in os.listdir(cls.img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_file_name = self.image_file_names[idx]
        image_path = os.path.join(self.img_dir, img_file_name)
        image = Image.open(image_path).convert("RGB")

        if self.color_transform is not None:
            image = self.color_transform(image)

        if self.annotation_type == "panoptic":
            mask, id_map = self.load_panoptic_mask(img_file_name)
            instance_mask = tv_tensors.Mask(mask)

        if "masks" in self.annotation_type:
            segmask = self._load_segmentation_mask(img_file_name)
            instance_mask, bboxes = None, None
        else:
            segmask = None

        if self.annotation_type == 'bboxes':
            bboxes, class_ids = self._load_bboxes(img_file_name, (image.size[1], image.size[0]))
            instance_mask = None
        else:
            bboxes, class_ids = None, None

        if self.annotation_type == "bboxes-detr":
            image_annos = self._read_bboxes_from_file(img_file_name)
            class_ids = [a["category_id"] for a in image_annos]
            class_ids = self._merge_classes(np.array(class_ids)).tolist()
            return {
                "image_id": idx,
                "image": image,
                "width": image.size[0],
                "height": image.size[1],
                "objects": {'id': [a["id"] for a in image_annos],
                            'area': [a["area"] for a in image_annos],
                            'bbox': [a["bbox"] for a in image_annos],
                            'category': class_ids,
                            }
            }

        if not self.no_transform:
            image, segmask, bboxes, instance_mask = self.transform(image, segmask, bboxes, instance_mask)

        if self.annotation_type == "masks":
            return image, segmask
        elif self.annotation_type == "bboxes":
            padding_mask = get_padding_mask(image)
            return image, padding_mask, bboxes, class_ids
        elif self.annotation_type == "masks+bboxes":
            return image, segmask, bboxes, class_ids
        elif self.annotation_type == "panoptic":
            padding_mask = get_padding_mask(image)
            return image, padding_mask, instance_mask, id_map

    def _get_segmentation_mask_from_rle(self, image_id: int) -> torch.Tensor:

        out_shape = self.ids_to_shapes[image_id]
        mask = np.zeros(out_shape, dtype=np.uint8)
        relevant_annotations = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        for class_id in range(1, 8):
            for ann in relevant_annotations:
                if ann['category_id'] == class_id:
                    rle = ann['segmentation']
                    binary_mask = mask_utils.decode(rle)
                    mask[binary_mask == 1] = class_id
        return mask

    def _get_panoptic_mask_from_rle(self, image_id: int) -> Tuple[torch.Tensor, dict]:
        out_shape = self.ids_to_shapes[image_id]
        mask = np.zeros(out_shape, dtype=np.int64)
        relevant_annotations = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        instance_to_class_map = {0: 0}
        for ann in relevant_annotations:
            instance_id = ann['id'] + 1
            class_id = ann['category_id']
            rle = ann['segmentation']
            binary_mask = mask_utils.decode(rle)
            mask[binary_mask == 1] = instance_id
            instance_to_class_map[instance_id] = class_id
        return mask, instance_to_class_map

    def load_panoptic_mask(self, image_filename: str) -> Tuple[torch.Tensor, dict]:
        image_id = self.filenames_to_ids[image_filename]
        mask, id_map = self._get_panoptic_mask_from_rle(image_id)

        # merge classes in dict
        for k, v in id_map.items():
            if not self.separate_background_banana and v == 2:
                id_map[k] = 1
            if not self.separate_defect_types and v > 2 and v != 255:
                id_map[k] = 3 if self.separate_background_banana else 2

        # optional preprocessing
        if self.preprocess_defects:
            for defect_id in set(id_map.values()) & set(self.defect_ids):
                defect_instance_ids = {k for k, v in id_map.items() if v == defect_id}
                # drop all defect instance_ids from id_map
                for instance_id in defect_instance_ids:
                    id_map.pop(instance_id)
                max_id = max(id_map.keys())

                defect_instance_mask = np.isin(mask, list(defect_instance_ids))
                separated_defects = self._separate_masks(defect_instance_mask, dilation_iters=8)
                for instance_id in np.unique(separated_defects):
                    if instance_id == 0:
                        continue
                    mask[separated_defects == instance_id] = max_id + instance_id
                    id_map[max_id + instance_id] = defect_id

        return mask, id_map

    def _load_segmentation_mask(self, image_filename: str) -> torch.Tensor:

        image_id = self.filenames_to_ids[image_filename]
        mask = self._get_segmentation_mask_from_rle(image_id)

        if self.defect_ignore_pad_px > 0:
            defect_binary = np.isin(mask, self.defect_indexes)
            dilated_defect_binary = binary_dilation(defect_binary, iterations=self.defect_ignore_pad_px)
            defect_edges = dilated_defect_binary & ~defect_binary
            mask[defect_edges] = self.ignore_index

        mask = self._merge_classes(mask)

        mask = tv_tensors.Mask(mask)

        return mask

    def _read_bboxes_from_file(self, image_filename: str):
        image_id = self.filenames_to_ids[image_filename]
        annotations = deepcopy(self.annotations)  # could be solved more elegantly
        relevant_annotations = [a for a in annotations['annotations'] if a['image_id'] == image_id and "bbox" in a]
        for a in relevant_annotations:
            a["area"] = mask_utils.decode(
                a["segmentation"]).sum()  # could be precalculated and added to json
        return relevant_annotations

    def _load_bboxes(self, image_filename: str, shape) -> Tuple[torch.Tensor, torch.Tensor]:
        relevant_annotations = self._read_bboxes_from_file(image_filename)

        annos = [torch.tensor(a['bbox']) for a in relevant_annotations]

        bboxes = torch.stack(annos) if annos else torch.empty((0, 4), dtype=torch.long)
        bboxes = torchvision.ops.box_convert(bboxes, 'xywh', 'xyxy')
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=shape)

        class_ids = torch.tensor([a['category_id'] for a in relevant_annotations], dtype=torch.long)

        class_ids = self._merge_classes(class_ids)

        return bboxes, class_ids
