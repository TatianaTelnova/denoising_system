import pathlib

import torch
import torchvision


def tensor_to_img(img_tensor: torch.Tensor, img_dir: str, img_filename: list):
    """
    Saves a tensor as an image
    :param img_tensor: (tensor) data
    :param img_dir: directory for saving
    :param img_filename: image filenames
    """
    assert int(img_tensor.shape[0]) == len(img_filename), "Error, {0} != {1}".format(int(img_tensor.shape[0]),
                                                                                     len(img_filename))
    img_dir_path = pathlib.Path(img_dir)
    if not img_dir_path.exists():
        img_dir_path.mkdir(parents=True)
    for i in range(img_tensor.size(0)):
        torchvision.utils.save_image(img_tensor[i], str(img_dir_path) + '/' + img_filename[i])
