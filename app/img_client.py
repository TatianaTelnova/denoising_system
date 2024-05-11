import os
import pathlib
import pickle

import torch
import torchvision
from PIL import Image

from preprocessing import get_img_transform


def save_img(img_filename: str, local_dir='D:/local_dir/'):
    path_local_dir = pathlib.Path(local_dir)
    if not path_local_dir.exists():
        path_local_dir.mkdir(parents=True)


def tensor_to_img(img_tensor: torch.Tensor, img_dir, img_filename):
    assert int(img_tensor.shape[0]) == len(img_filename), "Error, {0} != {1}".format(int(img_tensor.shape[0]),
                                                                                     len(img_filename))
    img_dir_path = pathlib.Path("D:/noise/data/adv/deepfool/" + img_dir)
    if not img_dir_path.exists():
        img_dir_path.mkdir(parents=True)
    for i in range(img_tensor.size(0)):
        torchvision.utils.save_image(img_tensor[i], str(img_dir_path) + '/' + img_filename[i])


def img_list_to_tensor(img_list: list[torch.Tensor]):
    return torch.stack(img_list, dim=0)
