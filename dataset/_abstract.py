import random

from sklearn.model_selection import train_test_split

from dataset.innoterra import InnoterraDataset


def load_datasets(dataset: str = "innoterra", seed: int = 42, fixed_splits_id=None,
                  train_defect_ignore_pad_px: int = 0,
                  val_defect_ignore_pad_px: int = 0,
                  train_mask_source="annotated",
                  val_mask_source="sam",
                  **kwargs
                  ):
    """Returns a train and validation dataset."""
    if dataset == 'innoterra':
        all_indexes = list(range(InnoterraDataset(**kwargs).n_samples_total()))

        if fixed_splits_id is None:
            train_indexes, val_indexes = train_test_split(all_indexes, test_size=0.2, random_state=seed)
        else:
            # divide all indexes in 5 equal parts
            assert fixed_splits_id < 5
            random.seed(seed)
            random.shuffle(all_indexes)
            val_indexes = all_indexes[fixed_splits_id::5]
            train_indexes = list(set(all_indexes) - set(val_indexes))

        train_dataset = InnoterraDataset(sample_ids=train_indexes, augment=True,
                                         color_augment=True, defect_ignore_pad_px=train_defect_ignore_pad_px,
                                         defect_mask_source=train_mask_source,
                                         **kwargs)
        val_dataset = InnoterraDataset(sample_ids=val_indexes, augment=False,
                                       color_augment=False, defect_ignore_pad_px=val_defect_ignore_pad_px,
                                       defect_mask_source=val_mask_source,
                                       **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    return train_dataset, val_dataset


if __name__ == '__main__':
    for _i in range(5):
        ds_t, ds_v = load_datasets("innoterra", 42, fixed_splits_id=_i)
        print(len(ds_t), len(ds_v))

        for t_img_name in ds_t.image_file_names:
            assert t_img_name not in ds_v.image_file_names