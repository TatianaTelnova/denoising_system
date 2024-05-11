from torch import tensor
from torchvision import transforms
from typing import Final

from PIL import Image

from model_autoencoder import Autoencoder

PREPROCESSING_TRANSFORM: Final = transforms.Compose([
    transforms.Resize(200),
    transforms.ToTensor()
])


def get_img_transform(img: Image) -> tensor:
    """
    Transforms an Image object into a tensor with size (3x200x200).
    :param img: Image object
    :return: tensor 200x200
    """
    return PREPROCESSING_TRANSFORM(img)


def get_tensor_by_img(img: Image) -> tensor:
    """
    Transforms an Image object into a tensor with size (1x3x200x200).
    :param img: Image object
    :return: tensor 1x3x200x200
    """
    return get_img_transform(img).unsqueeze(0)


def get_tensor_preprocess(img_tensor: tensor) -> tensor:
    """
    Preprocesses a tensor to delete noise.
    :param img_tensor: image tensor
    :return: de-noised image tensor
    """
    autoencoder = Autoencoder()
    return autoencoder(img_tensor)
