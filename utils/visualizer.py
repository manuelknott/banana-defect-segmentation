from typing import Union, List

import torch
import numpy as np
import matplotlib.pyplot as plt

DETAILED = {
    0: (0, 0, 0),  # background
    1: (255, 165, 0),  # background banana, orange
    2: (255, 255, 0),  # foreground banana, yellow
    3: (255, 0, 0),  # red
    4: (128, 0, 128),  # purple
    5: (255, 192, 203),  # pink
    6: (128, 0, 128),  # violet
    7: (0, 128, 128),  # teal
    255: (255, 255, 255),  # ignore index, white
}

COARSE_DEFECTS = {
    0: (0, 0, 0),  # background
    1: (255, 165, 0),  # background banana, orange
    2: (255, 255, 0),  # foreground banana, yellow
    3: (255, 0, 0),  # red
    4: (255, 0, 0),  # red
    5: (255, 0, 0),  # red
    6: (255, 0, 0),  # red
    7: (255, 0, 0),  # red
    255: (255, 255, 255),  # ignore index, white
}

THREE = {
    0: (0, 0, 0),  # background
    1: (255, 255, 0),  # foreground banana, yellow
    2: (255, 0, 0),  # red
    255: (255, 255, 255),  # ignore index, white
}


def convert_to_rgb_uint(color_name):
    # Get the RGB color in float format
    rgb_float = plt.cm.get_cmap("tab10")(color_name)[0:3]
    rgb_uint = tuple(int(c * 255) for c in rgb_float)
    return rgb_uint


GENERIC = {0: (0, 0, 0)} | {i: convert_to_rgb_uint(i - 1) for i in range(1, 10)}

PALLETES = {
    "detailed": DETAILED,
    "coarse": COARSE_DEFECTS,
    "generic": GENERIC,
    "random": None,
    "three": THREE
}


def create_random_pallette(ids: List[int]):
    colors = np.random.randint(0, 255, (len(ids), 3))
    pallette = {i: tuple(colors[e]) for e, i in enumerate(ids)}
    pallette[255] = (255, 255, 255)
    pallette[0] = (0, 0, 0)
    return pallette


class SegmentationMapVisualizer:

    def __init__(self, num_labels: int = 5, pallette="detailed"):  # TODO refactor num_labels argument

        self.random_pallette = pallette == "random"
        pallette = PALLETES[pallette]

        if num_labels == 3:  # To ensure the colors in the vizualizer are consistent across multiple runs
            keep = [0, 2, 3]
            self.palette = {k: pallette[k] for k in keep}
        elif num_labels == 4:
            keep = [0, 1, 2, 3]
            self.palette = {k: pallette[k] for k in keep}
        elif num_labels >= 5:
            self.palette = pallette

    def __call__(self, x: Union[torch.Tensor, np.ndarray]):

        if self.random_pallette:
            self.palette = create_random_pallette(np.unique(x))

        input_is_numpy = isinstance(x, np.ndarray)

        if input_is_numpy:
            x = torch.from_numpy(x)
        out = torch.zeros((3, x.shape[0], x.shape[1])).to(dtype=torch.uint8, device=x.device)
        for channel_id in range(3):
            for key, val in self.palette.items():
                mask = (x == key)
                out[channel_id][mask] = val[channel_id]

        if input_is_numpy:
            out = out.permute(1, 2, 0).numpy()
        return out


def make_axes_invisible(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


if __name__ == '__main__':
    vis = SegmentationMapVisualizer()
    img = torch.ones(3, 256, 256)
    out = vis(img)

    plt.imshow(out.permute(1, 2, 0))
    plt.show()
