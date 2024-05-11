import torch

import preprocessing

from dataset import AdvImgDataset, AdvDataset
from model_classification import ClassificationModel
from reporter import reporter, calculate_prediction


def example_clean_img():
    clean_dataset = {
        'val': AdvImgDataset(
            label_list=[0],
            img_list=['D:/noise/data/generated/val/bowl'],
            transform=preprocessing.PREPROCESSING_TRANSFORM)}
    img_labels, imd_data, _ = clean_dataset['val'].get_sample(2)
    model = ClassificationModel(is_custom_pretrained=True).model
    accuracy = calculate_prediction(prediction_model=model, label_tensor=img_labels, img_tensor=imd_data)
    print(accuracy)


def example_clean_vs_deepfool():
    # adv_deepfool_dataset = {
    #     'val': AdvDataset(
    #         label_list=[0],
    #         clean_list=['D:/noise/data/adv/deepfool_tensor/bowl/bowl_clean1.pckl'],
    #         adv_list=['D:/noise/data/adv/deepfool_tensor/bowl/bowl_adv.pckl'],
    #         filename_list=['D:/noise/data/adv/deepfool_tensor/bowl/bowl_label.pckl']
    #     )
    # }
    # img_labels, imd_data_clean, imd_data_adv, _ = adv_deepfool_dataset['val'].get_sample(2)
    # model = ClassificationModel(is_custom_pretrained=True).model
    # accuracy = calculate_prediction(prediction_model=model, label_tensor=torch.IntTensor(img_labels),
    #                                 img_tensor=imd_data_clean)
    # print(accuracy)
    # accuracy = calculate_prediction(prediction_model=model, label_tensor=torch.IntTensor(img_labels),
    #                                 img_tensor=imd_data_adv)
    # print(accuracy)
    reporter()


if __name__ == '__main__':
    # example_clean_img()
    example_clean_vs_deepfool()
