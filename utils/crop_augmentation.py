import matplotlib.pyplot as plt
import torch


def get_padding_mask(image: torch.Tensor):
    """returns a binary mask of the same size as the input image,
    where 1s represent the actual image and 0s represent padding
    (identified by full row/column black pixels")
    """

    is_row_black = (image.sum(dim=0).sum(dim=1) == 0)
    is_col_black = (image.sum(dim=0).sum(dim=0) == 0)

    # Calculate the indices where the rows and columns start to be non-black
    first_non_black_row = torch.where(~is_row_black)[0][0] if is_row_black.any() else 0
    last_non_black_row = torch.where(~is_row_black)[0][-1] if is_row_black.any() else image.shape[1] - 1
    first_non_black_col = torch.where(~is_col_black)[0][0] if is_col_black.any() else 0
    last_non_black_col = torch.where(~is_col_black)[0][-1] if is_col_black.any() else image.shape[2] - 1

    # Initialize a mask of zeros with the same height and width as the image
    mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.int32)

    # Update the mask to 1 for non-black areas
    mask[first_non_black_row:last_non_black_row + 1, first_non_black_col:last_non_black_col + 1] = 1
    return mask


if __name__ == '__main__':
    img = torch.randn(3, 444, 512)
    import torchvision.transforms as T

    img = T.CenterCrop((1024, 1024))(img)

    padding_mask = get_padding_mask(img)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print(padding_mask)
    plt.imshow(padding_mask, cmap='gray')
    plt.show()
