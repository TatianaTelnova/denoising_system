import matplotlib.pyplot as plt
import torch
from PIL import Image


def display_img_tensor(img_tensor: torch.Tensor, img_save: bool = False, img_filename: str = None):
    """
    Plots the image tensor and save it if img_save=True
    :param img_tensor:
    :param img_save:
    :param img_filename:
    """
    assert img_tensor.shape[0] == 3  # 3 channels (RGB)
    img_plt = torch.tensor(img_tensor).squeeze().permute(1, 2, 0)
    plt.imshow(img_plt)
    if img_save:
        if img_filename is None:
            img_filename = 'img.png'
        plt.savefig(img_filename, dpi=300, bbox_inches='tight')
    plt.show()


def display_img(image_path: str):
    """
    Plots image by image path
    :param image_path:
    """
    plt.imshow(Image.open(image_path, 'r'))
    plt.show()
