import os

from PIL import Image

from label import Label
from model_classification import ClassificationModel
from preprocessing import get_img_transform, get_tensor_preprocess


def get_classifier_label(img_filename: str, with_preprocessing: bool = True) -> str:
    """
    Returns the classifier label for the given image filename.
    """
    # проверка существования файла
    if os.path.isfile(img_filename):
        img_tensor = get_img_transform(Image.open(img_filename))
        # предобработка
        if with_preprocessing:
            img_tensor = get_tensor_preprocess(img_tensor)
        # классификация
        model = ClassificationModel(is_custom_pretrained=True).model
        out = model.predict(img_tensor)
        return Label(out).name
    else:
        print(f"File {img_filename} not found")
