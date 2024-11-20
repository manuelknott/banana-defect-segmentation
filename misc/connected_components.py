""" Demonstration of separate component algorithms to get masks for individual defects"""

import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_dilation

from utils.visualizer import SegmentationMapVisualizer

IMG_DIR = "/datasets/innoterra_segmentation/_old/resized_images"
MASK_DIR = "/datasets/innoterra_segmentation/_old/resized_masks_separate_defects"

ALL_IMGS = os.listdir(IMG_DIR)

FG_BANANA_ID = 2
BG_BANANA_ID = 1
DEFECT_IDS = [3, 4, 5, 6, 7]


def load_img_and_mask(idx: int):
    img_file = ALL_IMGS[idx]
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file.replace(".jpg", ".npy"))
    img = np.asarray(Image.open(img_path))
    mask = np.load(mask_path)
    return img, mask


def is_foreground_defect(defect_mask: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray, iters=1) -> bool:
    """
    Applies binary dilation to the defect mask and checks if it intersects with the foreground mask
    To address edge cases, we decide that a defect is foreground if the FG dilation overlap is bigger than BG dilation overlap.

    All arrays are expected to be binary (True, False)
    """
    dilated_mask = binary_dilation(defect_mask, iterations=iters)
    fg_overlap = np.sum(dilated_mask * fg_mask)
    bg_overlap = np.sum(dilated_mask * bg_mask)
    return fg_overlap > bg_overlap


def separate_masks(global_mask: np.ndarray, dilation_iters=4) -> np.ndarray:
    if dilation_iters == 0:
        defects_mask_uint = np.where(global_mask, 255, 0).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(defects_mask_uint)
        return labels_im

    defects_mask_dilated = binary_dilation(global_mask, iterations=dilation_iters)
    defects_mask_uint = np.where(defects_mask_dilated, 255, 0).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(defects_mask_uint)

    return np.where(global_mask, labels_im, 0)


def display_img_mask_components(idx: int, show=True):
    img, labels = load_img_and_mask(idx)

    segmap_vis = SegmentationMapVisualizer(pallette="detailed")(labels)
    instance_vis = SegmentationMapVisualizer(pallette="random")(labels)

    for defect_id in DEFECT_IDS:
        defect_mask = labels == defect_id
        defect_area = np.sum(defect_mask)
        if defect_area == 0:
            continue

        defect_instances = separate_masks(defect_mask)

        defects_vis = SegmentationMapVisualizer(pallette="generic")(defect_instances)

        # remove background defects
        fg_banana_mask = labels == FG_BANANA_ID
        bg_banana_mask = labels == BG_BANANA_ID
        defect_instance_ids = np.unique(defect_instances[defect_instances > 0])
        fg_defects = np.zeros_like(defect_instances)
        for defect_instance_id in defect_instance_ids:
            defect_mask = defect_instances == defect_instance_id
            if is_foreground_defect(defect_mask, fg_banana_mask, bg_banana_mask):
                fg_defects[defect_mask] = defect_instance_id

    fg_defects_vis = SegmentationMapVisualizer(pallette="generic")(fg_defects)

    n_fg_defects = len(np.unique(fg_defects[fg_defects > 0]))

    all_fg_defects_mask = fg_defects > 0
    affected_area = np.sum(all_fg_defects_mask) / np.sum(fg_banana_mask | all_fg_defects_mask)

    fig, ax = plt.subplots(1, 5, constrained_layout=True, figsize=(20, 5))
    ax[0].imshow(img)
    ax[1].imshow(segmap_vis)
    ax[2].imshow(instance_vis)

    ax[3].imshow(defects_vis)
    ax[4].imshow(fg_defects_vis)

    ax[1].set_title("Complete mask")
    ax[3].set_title(f"Separated defects\n(Connected Components Alg.)")
    ax[4].set_title(f"Foreground defects\n(Dilation filter)")

    fig.suptitle(f"N defects: {n_fg_defects}, Affected area: {affected_area * 100:.2f}%", fontsize=16)

    # deactivate all axes
    for a in ax:
        a.set_axis_off()

    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    img_id = 57
    for i in range(img_id, img_id+1):
        display_img_mask_components(i)
