# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine-tuning/Fine_tuning_MaskFormer_on_a_panoptic_dataset.ipynb
import argparse
import os
import pickle
from datetime import datetime
import yaml

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, OneFormerImageProcessor, \
    OneFormerForUniversalSegmentation
from torchmetrics.detection import PanopticQuality, MeanAveragePrecision
from pytorch_lightning import seed_everything

from dataset import load_datasets
from evaluation.semantic_seg_metrics import SemanticSegmentationMetrics
from utils.visualizer import SegmentationMapVisualizer
from postprocess import postprocess_list, postprocess

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--model", choices=["maskformer", "oneformer"], type=str, default="maskformer")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--split_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--eval_every_epochs", type=int, default=5)
parser.add_argument("--mask_source", type=str, choices=["annotated", "sam-b", "sam-l", "sam-h", "sam2-l"],
                    default="annotated")
parser.add_argument("--eval_anno", action="store_true")
parser.add_argument("--separate_defect_types", action="store_true")
parser.add_argument("--separate_background_banana", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--postprocess", action="store_true")
parser.add_argument("--eval_only", action="store_true")
args = parser.parse_args()

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

wandb.login(key=config['wandb']['key'])

log_dict = dict()
best_metric = 0.0

instance_visualizer = SegmentationMapVisualizer(pallette="random")
semantic_visualizer = SegmentationMapVisualizer(
    pallette="three" if not args.separate_background_banana else "detailed")

img_processor = MaskFormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=True,
                                         do_reduce_labels=False, image_mean=ADE_MEAN, image_std=ADE_STD)


def unnormalize(tensor, mean=ADE_MEAN, std=ADE_STD):
    tensor[0, :, :] = tensor[0, :, :] * std[0] + mean[0]
    tensor[1, :, :] = tensor[1, :, :] * std[1] + mean[1]
    tensor[2, :, :] = tensor[2, :, :] * std[2] + mean[2]
    return tensor


def binarymask2bbox(mask: torch.Tensor):
    nonzero_indices = torch.nonzero(mask)

    # Determine the smallest and largest row and column indices
    min_row = torch.min(nonzero_indices[:, 0])
    max_row = torch.max(nonzero_indices[:, 0])
    min_col = torch.min(nonzero_indices[:, 1])
    max_col = torch.max(nonzero_indices[:, 1])
    return torch.Tensor([min_col.item(), min_row.item(), max_col.item(), max_row.item()])


def convert_binary_masks(instance_binary_masks: torch.tensor, class_labels: torch.tensor):
    """Create semantic map / instance id map from binary masks"""
    assert instance_binary_masks.shape[0] == class_labels.shape[0]
    instance_mask = torch.zeros((1024, 1024))
    semantic_mask = torch.zeros((1024, 1024))
    for i, (bin_mask, class_id) in enumerate(zip(instance_binary_masks, class_labels)):
        instance_mask[bin_mask == 1] = i
        semantic_mask[bin_mask == 1] = class_id
    return instance_mask, semantic_mask


def set_trainable(model, pixel_level_module=False, transformer_module=False, class_predictor=True, mask_embedder=True):
    for n, p in model.named_parameters():
        if "pixel_level_module" in n:
            p.requires_grad = pixel_level_module
        elif "transformer_module" in n:
            p.requires_grad = transformer_module
        elif "class_predictor" in n:
            p.requires_grad = class_predictor
        elif "mask_embedder" in n:
            p.requires_grad = mask_embedder


def collate_fn(batch):
    pixel_vals = torch.stack([b[0] for b in batch])
    padding_masks = torch.stack([b[1] for b in batch])
    instance_masks = torch.stack([b[2] for b in batch])
    instance_to_semantic_maps = [b[3] for b in batch]

    inputs = img_processor(pixel_vals,
                           instance_masks,
                           instance_id_to_semantic_id=instance_to_semantic_maps,
                           return_tensors="pt")
    inputs["pixel_mask"] = padding_masks  # override pixel mask

    return inputs


def train_epoch():
    global log_dict
    model.train()
    running_loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch}")):
        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        # Optimization
        optimizer.step()

    log_dict["train_loss"] = running_loss / num_samples


@torch.no_grad()
def evaluate(model, dataloader, subset: str = "val", n_vis_imgs: int = 5):
    global log_dict
    global best_metric
    model.eval()

    semantic_metrics = SemanticSegmentationMetrics(classnames=train_dataset.class_names)
    panoptic_quality = PanopticQuality(things=train_dataset.defect_ids, stuffs={0} | train_dataset.banana_ids,
                                       allow_unknown_preds_category=True)
    detection_metrics = MeanAveragePrecision(class_metrics=True, iou_type="bbox")

    assert subset in ["train", "val"]
    # dataloader = train_dataloader if subset == "train" else val_dataloader

    running_loss = 0.0
    num_samples = 0

    all_results = []

    for batch in tqdm(dataloader, desc=f"Eval Epoch {epoch} ({subset})"):

        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        loss = outputs.loss.item()
        batch_size = batch["pixel_values"].size(0)
        running_loss += loss

        # handle predictions
        results = img_processor.post_process_panoptic_segmentation(outputs,
                                                                   label_ids_to_fuse={0} | train_dataset.banana_ids,
                                                                   target_sizes=[(1024, 1024) for _ in
                                                                                 range(batch_size)])

        if args.postprocess:
            results = postprocess_list(results)

        instance2class_maps = [{s["id"]: s["label_id"] for s in r["segments_info"]} for r in results]
        instance2class_maps = [{0: 0, -1: -1, **m} for m in instance2class_maps]

        all_results.extend(results)

        for img_id in range(batch_size):
            gt_instance_binary_masks = batch["mask_labels"][img_id]
            gt_class_labels = batch["class_labels"][img_id]
            gt_instance_mask, gt_semantic_mask = convert_binary_masks(gt_instance_binary_masks, gt_class_labels)
            gt_combined_mask = torch.stack([gt_semantic_mask, gt_instance_mask], dim=-1).detach().cpu().to(torch.long)

            pred_instance_mask = results[img_id]["segmentation"]
            pred_semantic_mask = pred_instance_mask.clone()
            for instance_id, class_id in instance2class_maps[img_id].items():
                pred_semantic_mask[pred_instance_mask == instance_id] = class_id
            pred_combined_mask = torch.stack([pred_semantic_mask, pred_instance_mask], dim=-1).detach().cpu().to(
                torch.long)

            # PANOPTIC METRICS
            panoptic_quality.update(pred_combined_mask.unsqueeze(0), gt_combined_mask.unsqueeze(0))

            # SEMANTIC METRICS
            semantic_metrics.update(pred_semantic_mask.cpu().to(torch.long),
                                    gt_semantic_mask.cpu().to(torch.long))

            ########################### OBJ DET METRICS
            _pred_boxes = list()
            _pred_masks = list()
            _pred_labels = list()
            _pred_scores = list()
            result = results[img_id]
            for segment_info in result["segments_info"]:
                if segment_info["label_id"] not in train_dataset.defect_ids:
                    continue
                _pred_mask = (result["segmentation"] == segment_info["id"]).detach().cpu()
                _pred_masks.append(_pred_mask)
                _pred_bbox = binarymask2bbox(_pred_mask)
                _pred_boxes.append(_pred_bbox)
                _pred_labels.append(
                    torch.LongTensor([segment_info["label_id"]]) - (3 if args.separate_background_banana else 2))
                _pred_scores.append(torch.FloatTensor([segment_info["score"]]))
            detection_preds = [{
                "boxes": torch.stack(_pred_boxes) if len(_pred_boxes) > 0 else torch.zeros((0, 4)),
                "masks": torch.stack(_pred_masks) if len(_pred_masks) > 0 else torch.zeros((0, 1024, 1024)),
                "labels": torch.cat(_pred_labels) if len(_pred_labels) > 0 else torch.LongTensor([]),
                "scores": torch.cat(_pred_scores) if len(_pred_scores) > 0 else torch.FloatTensor([])
            }]

            _gt_boxes = list()
            _gt_masks = list()
            _gt_labels = list()
            for cid, class_label in enumerate(batch["class_labels"][img_id]):
                if class_label.item() not in train_dataset.defect_ids:
                    continue
                gt_mask = (batch["mask_labels"][img_id]).detach().cpu().to(torch.bool)[cid]
                _gt_masks.append(gt_mask)
                gt_bbox = binarymask2bbox(gt_mask)
                _gt_boxes.append(gt_bbox)
                _gt_labels.append(torch.LongTensor([class_label]) - (3 if args.separate_background_banana else 2))
            detection_targets = [{
                "boxes": torch.stack(_gt_boxes) if len(_gt_boxes) > 0 else torch.zeros((0, 4)),
                "masks": torch.stack(_gt_masks) if len(_gt_masks) > 0 else torch.zeros((0, 1024, 1024)),
                "labels": torch.cat(_gt_labels) if len(_gt_labels) > 0 else torch.LongTensor([])
            }]

            detection_metrics.update(detection_preds, detection_targets)
            ###########################

            # visualize
            if n_vis_imgs != -1 and num_samples < n_vis_imgs:

                # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                # img = batch["pixel_values"][img_id]
                # ax[0][0].imshow(unnormalize(img).permute(1, 2, 0).cpu())
                # ax[0][1].imshow(semantic_visualizer(gt_semantic_mask.detach().cpu()).permute(1, 2, 0))
                # ax[1][0].imshow(instance_visualizer(pred_instance_mask.detach().cpu()).permute(1, 2, 0))
                # ax[1][1].imshow(semantic_visualizer(pred_semantic_mask.detach().cpu()).permute(1, 2, 0))

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                img = batch["pixel_values"][img_id]
                ax[0].imshow(unnormalize(img).permute(1, 2, 0).cpu())
                ax[1].imshow(semantic_visualizer(gt_semantic_mask.detach().cpu()).permute(1, 2, 0))
                ax[2].imshow(semantic_visualizer(pred_semantic_mask.detach().cpu()).permute(1, 2, 0))
                # ax[3].imshow(instance_visualizer(pred_instance_mask.detach().cpu()).permute(1, 2, 0))
                # plot bounding boxes around ax[1]
                for box, label in zip(_gt_boxes, _gt_labels):
                    x1, y1, x2, y2 = box
                    class_id = label.item() + (3 if args.separate_background_banana else 2)
                    _color = semantic_visualizer.palette[class_id]
                    _color = [c / 255 for c in _color]
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=_color, facecolor='none')
                    ax[1].add_patch(rect)

                # plot bounding boxes around ax[2]
                for box, label in zip(_pred_boxes, _pred_labels):
                    x1, y1, x2, y2 = box
                    class_id = label.item() + (3 if args.separate_background_banana else 2)
                    _color = semantic_visualizer.palette[class_id]
                    _color = [c / 255 for c in _color]
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=_color, facecolor='none')
                    ax[2].add_patch(rect)

                for _ax in ax.ravel():
                    _ax.axis("off")
                plt.tight_layout()
                samples_dir = f"samples/{run_name}{'_pp' if args.postprocess else ''}{'_evalanno' if args.eval_anno else ''}"
                os.makedirs(samples_dir, exist_ok=True)
                plt.savefig(f"{samples_dir}/{num_samples}.png")
                if args.debug:
                    plt.show()
                log_dict[f"{subset}_visualizations_{num_samples}"] = wandb.Image(fig, caption=f"img_{num_samples}")
                plt.close()

            num_samples += 1

    final_loss = running_loss / num_samples
    pq = panoptic_quality.compute().item()
    semantic_results = semantic_metrics.compute()
    log_dict[f"val_loss"] = final_loss
    log_dict[f"val_panoptic_quality"] = pq
    for n, v in semantic_results.items():
        log_dict[f"val_{n}"] = v.item()

    detection_results = detection_metrics.compute()

    for metric in ["map", "map_50", "map_75", "mar_1", "mar_10", "mar_100"]:
        _metric_val = detection_results[metric].item()
        log_dict[f"val_{metric}"] = _metric_val if _metric_val != -1. else 0.
    classes = detection_results["classes"].reshape(-1)
    map_per_class = detection_results["map_per_class"].reshape(-1)
    mar_100_per_class = detection_results["mar_100_per_class"].reshape(-1)
    for _class, _class_map, _class_mar100 in zip(classes, map_per_class, mar_100_per_class):
        log_dict[f"val_map_{_class}"] = _class_map if _class_map != -1. else 0.
        log_dict[f"val_mar100_{_class}"] = _class_mar100 if _class_mar100 != -1. else 0.

    # save eval results to file
    with open(
            f"{config['ckpt_dir']}/{run_name}/eval_results{'_pp' if args.postprocess else ''}{'_evalanno' if args.eval_anno else ''}.pkl",
            "wb") as f:
        pickle.dump(log_dict, f)

    wandb.log(log_dict)
    print(log_dict)
    log_dict = dict()

    if pq > best_metric and not args.eval_only:
        best_metric = pq
        print(f"saving new best model with PQ={pq} at epoch {epoch}")
        model.save_pretrained(f"{config['ckpt_dir']}/{run_name}")

    # save all results to pickle
    with open(
            f"{config['ckpt_dir']}/{run_name}/predictions{'_pp' if args.postprocess else ''}{'_evalanno' if args.eval_anno else ''}.pkl",
            "wb") as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':

    seed_everything(args.seed)

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name if args.run_name is not None else f"maskformer_split{args.split_id}"

    if args.mask_source.startswith("sam"):
        run_name += f"_{args.mask_source}"

    if args.separate_background_banana:
        run_name += "_bg"

    if args.separate_defect_types:
        run_name += "_defects"

    print(f"======Starting run {run_name}=======")
    os.makedirs(f"{config['ckpt_dir']}/{run_name}", exist_ok=True)

    if not args.eval_only:
        wandb_run = wandb.init(project=config["wandb"]["project"],
                               group=args.model,
                               name=run_name,
                               config=vars(args),
                               entity=config["wandb"]["entity"],
                               mode="offline" if args.debug else "online"
                               )
    else:
        wandb.init(mode="disabled")

    train_dataset, val_dataset = load_datasets("innoterra", seed=args.seed, fixed_splits_id=args.split_id,
                                               annotation_type="panoptic",
                                               separate_background_banana=args.separate_background_banana,
                                               separate_defect_types=args.separate_defect_types,
                                               preprocess_defects=not args.separate_defect_types,
                                               train_mask_source=args.mask_source,
                                               val_mask_source="annotated" if args.eval_anno else args.mask_source,
                                               )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    ckpt_path = f"{config['ckpt_dir']}/{run_name}" if args.eval_only else "facebook/maskformer-swin-base-ade"
    model = MaskFormerForInstanceSegmentation.from_pretrained(ckpt_path,
                                                              id2label=train_dataset.class_dict,
                                                              ignore_mismatched_sizes=True)
    set_trainable(model, pixel_level_module=True, transformer_module=True, class_predictor=True, mask_embedder=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    if not args.eval_only:
        for epoch in range(args.epochs):
            train_epoch()
            if epoch % args.eval_every_epochs == 0:
                evaluate(model, val_dataloader, "val", n_vis_imgs=-1)
            else:
                log_dict = dict()

    print("Evaluating model on validation set")
    epoch = 0
    if not args.eval_only:
        #  reload best model
        print("Reloading best model for eval")
        ckpt_path = f"{config['ckpt_dir']}/{run_name}"
        model = MaskFormerForInstanceSegmentation.from_pretrained(ckpt_path,
                                                                  id2label=train_dataset.class_dict,
                                                                  ignore_mismatched_sizes=True)
        model.to(device)

    evaluate(model, val_dataloader, "val", n_vis_imgs=-1)
