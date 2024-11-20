from typing import List

import numpy as np
import segmentation_models_pytorch as smp
import torch


class SemanticSegmentationMetrics:

    def __init__(self, classnames: List[str], ignore_index: int = 255):
        self.classnames = classnames
        self.n_classes = len(classnames)
        self.ignore_index = ignore_index

        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []

    def reset(self):
        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []

    def update(self, pred_mask: torch.LongTensor, target_mask: torch.LongTensor):
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask,
                                               target_mask,
                                               mode="multiclass",
                                               num_classes=self.n_classes,
                                               ignore_index=self.ignore_index)

        self.tp.append(tp.sum(dim=0))
        self.fp.append(fp.sum(dim=0))
        self.fn.append(fn.sum(dim=0))
        self.tn.append(tn.sum(dim=0))

    def compute(self):
        results = dict()

        tp = torch.stack(self.tp).sum(dim=0)
        fp = torch.stack(self.fp).sum(dim=0)
        fn = torch.stack(self.fn).sum(dim=0)
        tn = torch.stack(self.tn).sum(dim=0)

        iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        f1_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)

        for cls_name, _miou, _f1 in zip(self.classnames, iou_per_class, f1_per_class):
            results[f"IoU_'{cls_name}'"] = _miou
            results[f"F1_'{cls_name}'"] = _f1

        mean_iou = iou_per_class.mean()
        mean_f1 = f1_per_class.mean()
        results["mIoU"] = mean_iou
        results["mF1"] = mean_f1

        return results
