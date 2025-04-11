# Weakly Supervised Panoptic Segmentation for Banana Defects

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2411.16219)

This repository contains supplementary code for the paper [**Weakly supervised panoptic segmentation for defect-based grading of fresh produce**](http://arxiv.org/abs/2411.16219).
Accepted as a paper to **Agriculture-Vision: Challenges & Opportunities for Computer Vision in Agriculture (CVPR 2025 Workshops)**.

## Setup

This repository was tested using Python 3.11 and Pytorch 2.4.1.

### Install dependencies

```bash
pip install -r requirements.txt
```

### SAM2

If you want to use SAM2, run the following commands in addition:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
```

## Model training

First, set your W&B API key for logging and the dataset path in ´config.yaml´.

To train the model use the script `train_panoptic.py`, with the following options:

- `--run_name`: what run name to log
- `--model`: which panoptic segmentation model to use, currently only `maskformer` is supported
- `--seed`: integer to set random seed for reproducibility
- `--split_id`: which split to use for validation (cross-validation)
- `--batch_size`: batch size for training, defaults to 2
- `--epochs`: number of epochs to train for, defaults to 100
- `--eval_every_epochs`: how often to evaluate the model on the validation set
- `--mask_source`: which mask targets to use, either `annotated`, `sam` or `sam2`. SAM masks must be precalculated. See `misc/generate_sam_annotations.py`.
- `--eval_anno`: whether to evaluate the model on the annotated masks
- `--separate_defect_types`: If true, uses four defect categories instead of one
- `--separate_background_banana`: If true, uses two classes for bananas (foreground and background)
- `--postprocess`: If true, applies postprocessing for final defect masks (see paper for explanation)
- `--eval_only`: If true, tries to load a pretrained model and only runs eval on the validation splot


All results reported in the paper can be reproduced using the `train_all.sh` script analysis scripts in the `analysis` folder.

## Using a pre-trained model

To use a pre-trained model, download checkpoints [here](https://drive.google.com/file/d/1OS8G62eCMR4aN3-gVyY3w8WpoXFQ0qT0/view?usp=sharing) and run the following python code and pick a checkpoint.

Checkpoints are named after the following pattern:

If a checkpoint folder name contains `sam`/`sam2` it means that the model was trained using masks from these models.
To achieve the best results, use the same mask source as the model was trained on.
The `bg` suffix indicates that the model was trained with a separate background class for bananas.
The `defects` suffix indicates that the model was trained with four separate defect classes instead of a single one.

To run a model, use the following code, replacing `<path-to-checkpoint>` with the path to the downloaded checkpoint folder:

```python
    import matplotlib.pyplot as plt
    from inference import BananaSegmentationModel
    from utils.visualizer import SegmentationMapVisualizer

    # initialize model
    model = BananaSegmentationModel("<path-to-checkpoint>")

    # run inference on example image
    instance_mask, semantic_mask, segments_info = model.predict("example.jpg")

    # visualize results
    visualizer = SegmentationMapVisualizer()
    semantic_image = visualizer(semantic_mask)
    plt.imshow(semantic_image.permute(1, 2, 0))
    plt.show()
```

## Citation

If you find this project useful, please consider citing our preprint:
```
@article{knott2024weakly,
  title={Weakly supervised panoptic segmentation for defect-based grading of fresh produce}, 
  author={Manuel Knott and Divinefavour Odion and Sameer Sontakke and Anup Karwa and Thijs Defraeye}
  journal={arXiv preprint arXiv:2411.16219},
  year={2024}
}
```
