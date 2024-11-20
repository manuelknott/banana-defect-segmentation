import numpy as np
import matplotlib.pyplot as plt
import PIL

from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from utils.visualizer import SegmentationMapVisualizer

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class_dict_single = {
    0: "background",
    1: "background_banana",
    2: "foreground_banana",
    3: "defect"
}

class_dict_bg_single = {
    0: "background",
    1: "background_banana",
    2: "foreground_banana",
    3: "defect"
}

class_dict_multi = {
    0: "background",
    1: "background_banana",
    2: "foreground_banana",
    3: "old bruise",
    4: "old scar",
    5: "new bruise",
    6: "new scar",
}

class_dict_bg_multi = {
    0: "background",
    1: "background_banana",
    2: "foreground_banana",
    3: "old bruise",
    4: "old scar",
    5: "new bruise",
    6: "new scar",
}

class BananaSegmentationModel:

    def __init__(self, checkpoint_path: str,
                 use_background_banana: bool = True,
                 use_single_defect: bool = True,
                 ):

        if use_background_banana:
            self.non_defect_ids = {0, 1, 2}
        else:
            self.non_defect_ids = {0, 1}

        if not use_background_banana and use_single_defect:
            class_dict = class_dict_single
        elif use_background_banana and use_single_defect:
            class_dict = class_dict_bg_single
        elif not use_background_banana and not use_single_defect:
            class_dict = class_dict_multi
        else:
            class_dict = class_dict_bg_multi

        self.model = MaskFormerForInstanceSegmentation.from_pretrained(checkpoint_path,
                                                                  id2label=class_dict,
                                                                  ignore_mismatched_sizes=True)
        self.model.eval()
        self.img_processor = MaskFormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=True,
                                                 do_reduce_labels=False, image_mean=ADE_MEAN, image_std=ADE_STD)

    @staticmethod
    def create_semantic_mask(instance_mask, segments_info):
        instance2class_map = {s["id"]: s["label_id"] for s in segments_info}
        instance2class_map = {0: 0, -1: -1, **instance2class_map}

        semantic_mask = instance_mask.clone()
        for instance_id, class_id in instance2class_map.items():
            semantic_mask[instance_mask == instance_id] = class_id

        return semantic_mask


    def predict(self, image_path: str):
        img = PIL.Image.open(image_path)
        w, h = img.size
        inputs = self.img_processor(img, return_tensors="pt")
        outputs = self.model(**inputs)
        results = self.img_processor.post_process_panoptic_segmentation(outputs,
                                                                   label_ids_to_fuse=self.non_defect_ids,
                                                                   target_sizes=[(h, w)])

        instance_mask = results[0]["segmentation"]
        segments_info = results[0]["segments_info"]
        semantic_mask = self.create_semantic_mask(instance_mask, segments_info)

        return instance_mask, semantic_mask, segments_info




if __name__ == '__main__':
    model = BananaSegmentationModel("/mnt/hdd-4t/bananasam_checkpoints/ckpts/maskformer_split0_sam2_bg")

    instance_mask, semantic_mask, segments_info = model.predict("example.jpg")

    visualizer = SegmentationMapVisualizer()

    semantic_image = visualizer(semantic_mask)

    plt.imshow(semantic_image.permute(1, 2, 0))
    plt.show()
