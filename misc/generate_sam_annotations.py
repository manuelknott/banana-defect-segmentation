import os
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as maskUtils

from sam2.sam2_image_predictor import SAM2ImagePredictor

verbose = False

ANALYSE_IOU_ONLY =False

SAM_CKPTS = {
    "vit_b": "models/custom_semantic_sam/ckpt_cache/sam_vit_b_01ec64.pth",
    "vit_l": "models/custom_semantic_sam/ckpt_cache/sam_vit_l_0b3195.pth",
    "vit_h": "models/custom_semantic_sam/ckpt_cache/sam_vit_h_4b8939.pth",
}


def get_iou(binary_array1, binary_array2):
    # Ensure the arrays are boolean
    binary_array1 = np.asarray(binary_array1, dtype=bool)
    binary_array2 = np.asarray(binary_array2, dtype=bool)

    # Compute the intersection (logical AND)
    intersection = np.sum(np.logical_and(binary_array1, binary_array2))

    # Compute the union (logical OR)
    union = np.sum(np.logical_or(binary_array1, binary_array2))

    # Compute the IoU
    iou_score = intersection / union

    return iou_score

annotations = json.load(open('/datasets/innoterra_segmentation/annotations.json'))

def main(use_sam2:bool, sam_variant: str):
    if use_sam2:
        sam_model = sam_model_registry[f"vit_{sam_variant}"](checkpoint=SAM_CKPTS[f"vit_{sam_variant}"])
        sam = SamPredictor(sam_model)
    else:
        sam = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    img_id2path = {ann["id"]: ann["file_name"] for ann in annotations["images"]}

    #def rle_list_to_string(rle_list):
    #    return ' '.join(map(str, rle_list))



    all_iou = []

    for i in tqdm(range(len(annotations["annotations"]))):
        anno = annotations["annotations"][i]
        image_id = anno["image_id"]
        image_path = os.path.join("/datasets/innoterra_segmentation/resized_images_1024", img_id2path[image_id])

        bbox = anno.get("bbox")

        if bbox is None:
            continue # non-defect

        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h

        image = Image.open(image_path).convert("RGB")

        sam.set_image(np.array(image))
        masks, mask_qualities, _ = sam.predict(box=np.array([x1, y1, x2, y2]),
                                               multimask_output=True)
        best_sam_mask = masks[mask_qualities.argmax()]

        # compare sam mask with anno mask
        # first convert rle to binary
        anno_mask = maskUtils.decode(anno["segmentation"]).astype(np.bool_)
        # get iou
        iou = get_iou(anno_mask, best_sam_mask)
        all_iou.append(iou)

        if verbose:
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(anno_mask)
            ax[1].imshow(image)
            rect = Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
            ax[2].imshow(best_sam_mask)
            plt.suptitle(f"iou: {iou:.2f}")
            plt.tight_layout()
            plt.show()

        # convert to RLE
        best_sam_mask = best_sam_mask.astype(np.uint8)
        rle = (maskUtils.encode(np.asfortranarray(best_sam_mask)))
        rle["counts"] = rle["counts"].decode('utf-8')

        #print(annotations["annotations"][i]["segmentation"])
        annotations["annotations"][i]["segmentation"] = rle
        #print(annotations["annotations"][i]["segmentation"])

    # save all_iou to file
    print(all_iou)
    with open(f"analysis/{'sam2' if use_sam2 else 'sam'}_vit_{sam_variant}_iou.txt", "w") as f:
        for iou in all_iou:
            f.write(f"{iou}\n")

    # SAVE JSON
    if not ANALYSE_IOU_ONLY:
        with open(f"/datasets/innoterra_segmentation/annotations_{'sam2' if use_sam2 else 'sam'}_vit_{sam_variant}.json", "w") as f:
            json.dump(annotations, f)
            print(f"Saved to /datasets/innoterra_segmentation/annotations_{'sam2' if use_sam2 else 'sam'}_vit_{sam_variant}.json")

if __name__ == '__main__':
    #main(use_sam2=False, sam_variant="b")
    #main(use_sam2=False, sam_variant="l")
    main(use_sam2=False, sam_variant="h")
    #main(use_sam2=True, sam_variant="l")