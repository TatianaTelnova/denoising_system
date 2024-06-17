import torch

from torch.nn import Linear
from torchvision import models
from torchvision.models import VGG16_Weights


class ClassificationModel:
    """
    This class is used to create a classification model.
    """

    def __init__(self, num_classes: int, is_custom_pretrained: bool = True):
        """
        This method is used to create a classification model with or without pre-trained weights.
        :param is_custom_pretrained: True if the model is to be created with custom pre-trained weights, otherwise it
        will be created with the VGG16 pre-trained weights
        """
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier[6] = Linear(4096, num_classes)
        if is_custom_pretrained:
            model.load_state_dict(torch.load('model/model_210324.pth'))
        self.model = model.to('cpu').eval()
