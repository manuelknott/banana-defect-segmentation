from typing import Tuple, List, Dict
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_dilation

from utils.visualizer import SegmentationMapVisualizer


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


def postprocess(segmentations: torch.Tensor,
                segments_info: dict,
                has_background_banana: bool=True,
                remove_background_defects: bool=True,
                dilation_iters: int=8,
                min_pixels: int=50,
                ) -> Tuple[torch.Tensor, list]:


    instance2class = {s["id"]: s["label_id"] for s in segments_info}
    instance2class = {0: 0, -1: -1, **instance2class}
    non_defect_ids = {-1, 0, 1}
    if has_background_banana:
        non_defect_ids.add(2)

    orig_device = segmentations.device
    new_segmentations = segmentations.clone().to("cpu").numpy()

    # Reconnect components
    semantic_map = np.vectorize(instance2class.get)(new_segmentations)
    unique_defect_ids = [d for d in np.unique(new_segmentations).tolist() if d not in non_defect_ids]

    new_segments_info = [s for s in segments_info if s["label_id"] in non_defect_ids]
    new_segment_running_id = max([s["id"] for s in new_segments_info]) + 1

    for defect_id in unique_defect_ids:
        defect_mask = semantic_map == defect_id
        defect_area = defect_mask.sum().item()
        if defect_area < min_pixels:
            continue

        defect_instances = separate_masks(defect_mask, dilation_iters=dilation_iters)

        unique_defect_instance_ids = [udi for udi in np.unique(defect_instances) if udi > 0]
        for defect_instance_id in unique_defect_instance_ids:
            defect_instance_mask = defect_instances == defect_instance_id

            # plt.imshow(defect_mask)
            # plt.show()

            if defect_instance_mask.sum().item() < min_pixels:
                new_segmentations[defect_instance_mask] = 0
                #print("segment too small")
                continue

            # optional: remove background defects
            if has_background_banana and remove_background_defects:
                bg_banana_mask = semantic_map == 1
                fg_banana_mask = semantic_map == 2
                if not is_foreground_defect(defect_instance_mask, fg_banana_mask, bg_banana_mask):
                    new_segmentations[defect_instance_mask] = 0
                    #print("background defect")
                    continue

            # apply the defect mask
            new_segmentations[defect_instance_mask] = new_segment_running_id
            new_segments_info.append({'id': new_segment_running_id,
                                      'label_id': defect_id,
                                      'score': 1.0, # TODO pretty sure this has no effect
                                      'was_fused': False,
                                      })
            new_segment_running_id += 1

    new_segmentations = torch.from_numpy(new_segmentations).to(orig_device)
    return new_segmentations, new_segments_info


def postprocess_list(results: List[Dict]):
    out = []
    for res in results:
        seg, seg_info = postprocess(res["segmentation"], res["segments_info"])
        out.append({"segmentation": seg, "segments_info": seg_info})
    return out


if __name__ == '__main__':

    img = torch.load("sample_img.torch").cpu()
    res = pickle.load(open("sample_result.pkl", "rb"))

    seg = res["segmentation"].cpu()
    seg_info = res["segments_info"]

    new_seg, new_seg_info = postprocess(seg, seg_info, has_background_banana=True, remove_background_defects=True)

    old_instance2class = {s["id"]: s["label_id"] for s in seg_info}
    old_instance2class = {0: 0, -1: -1, **old_instance2class}

    new_instance2class = {s["id"]: s["label_id"] for s in new_seg_info}
    new_instance2class = {0: 0, -1: -1, **new_instance2class}

    segmap_vis = SegmentationMapVisualizer(pallette="detailed")
    instance_vis = SegmentationMapVisualizer(pallette="random")

    old_semantic_mask = seg.clone().apply_(old_instance2class.get)
    new_semantic_mask = new_seg.clone().apply_(new_instance2class.get)

    fig, ax = plt.subplots(1, 5, constrained_layout=True, figsize=(20, 5))
    ax[0].imshow(img)
    ax[1].imshow(segmap_vis(old_semantic_mask).permute(1,2,0))
    ax[2].imshow(instance_vis(seg).permute(1,2,0))
    ax[3].imshow(segmap_vis(new_semantic_mask).permute(1,2,0))
    ax[4].imshow(instance_vis(new_seg).permute(1,2,0))

    # deactivate all axes
    for a in ax:
        a.set_axis_off()

    plt.show()