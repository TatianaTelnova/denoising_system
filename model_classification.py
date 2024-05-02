import torch
from torchvision import models
from torchvision.models import VGG16_Weights


class ClassificationModel:
    """
    This class is used to create a classification model.
    """

    def __init__(self, is_custom_pretrained: bool = True):
        """
        This method is used to create a classification model with or without pre-trained weights.
        :param is_custom_pretrained: True if the model is to be created with custom pre-trained weights, otherwise it
        will be created with the VGG16 pre-trained weights
        """
        if is_custom_pretrained:
            weights = torch.load('D:/noise/model_210324.pth')
        else:
            weights = VGG16_Weights.IMAGENET1K_V1
        self.model = models.vgg16(weights=weights).to('cpu').eval()
