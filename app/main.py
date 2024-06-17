import enum

import torch
from foolbox.attacks import L2FastGradientAttack

from app import preprocessing
from app.classification import calculate_adv_prediction_multi_with_dump
from app.dataset import AdvImgDataset, AdvDataset
from app.denoising import get_noise_dataset, get_denoising_dataset
from app.label import Label
from app.model_autoencoder import AutoencoderModel
from app.model_classification import ClassificationModel
from app.reporter import reporter

DIR_PATH = 'D:/data'


def example_create_adv_dataset():
    # создаем датасет из изображений без шума
    label_list = [Label.Bowl, Label.Cat]
    clean_dataset = AdvImgDataset(
        label_list=[lbl.value for lbl in label_list],
        img_list=[DIR_PATH + lbl.name.lower() for lbl in label_list],
        transform=preprocessing.PREPROCESSING_TRANSFORM)
    # создаем модель сети классификации
    classification_model = ClassificationModel(num_classes=7, is_custom_pretrained=True).model
    # накладываем шум на изображения
    # данные сохранятся в файлы '<class_name>_clean.pkl', '<class_name>_clipped.pkl', '<class_name>_label.pkl'
    return get_noise_dataset(classification_model, L2FastGradientAttack(), clean_dataset)


def example_calculate_adv_prediction():
    # if __name__ == '__main__':
    label_list: list[enum.IntEnum] = [Label.Bowl, Label.Cat]
    # формируем датасет из изображений с шумом на основе файлов с данными
    adv_dataset = AdvDataset(
        label_list=[lbl.value for lbl in label_list],
        clean_list=[DIR_PATH + lbl.name.lower() + '_clean.pkl' for lbl in label_list],
        adv_list=[DIR_PATH + lbl.name.lower() + '_clipped.pkl' for lbl in label_list],
        filename_list=[DIR_PATH + lbl.name.lower() + '_label.pkl' for lbl in label_list])
    # создаем модель сети классификации
    classification_model = ClassificationModel(num_classes=7, is_custom_pretrained=True).model
    all_adv_data = adv_dataset.get_all()
    # вычисляем Accuracy с сохранением промежуточных итогов в файл './temp_prediction/prediction_.json'
    acc_clean, acc_adv = calculate_adv_prediction_multi_with_dump(classification_model,
                                                                  torch.IntTensor(all_adv_data[0]),
                                                                  {"clean": all_adv_data[1],
                                                                   "deepfool": all_adv_data[2]})
    print(f'Accuracy clean: {acc_clean}, Accuracy adv: {acc_adv}')
    return acc_clean, acc_adv


def example_calculate_adv_prediction_with_denoising():
    label_list = [Label.Bowl, Label.Cat]
    # формируем датасет из изображений с шумом на основе файлов с данными
    adv_dataset = AdvDataset(
        label_list=[lbl.value for lbl in label_list],
        clean_list=[DIR_PATH + lbl.name.lower() + '_clean.pkl' for lbl in label_list],
        adv_list=[DIR_PATH + lbl.name.lower() + '_clipped.pkl' for lbl in label_list],
        filename_list=[DIR_PATH + lbl.name.lower() + '_label.pkl' for lbl in label_list])
    all_adv_data = adv_dataset.get_all()
    denoising_model = AutoencoderModel(is_custom_pretrained=True).autoencoder
    classification_model = ClassificationModel(num_classes=7, is_custom_pretrained=True).model
    acc_list = calculate_adv_prediction_multi_with_dump(classification_model,
                                                        torch.IntTensor(all_adv_data[0]),
                                                        {"clean": all_adv_data[1],
                                                         "adv": all_adv_data[2],
                                                         "denoising": denoising_model(all_adv_data[2])})
    for acc, case_name in zip(acc_list, ["clean", "adv", "denoising"]):
        print(f'Accuracy {case_name}: {acc}')
    reporter()


def example_denoise_dataset():
    label_list = [Label.Bowl, Label.Cat]
    # формируем датасет из изображений с шумом на основе файлов с данными
    adv_dataset = AdvDataset(
        label_list=[lbl.value for lbl in label_list],
        clean_list=[DIR_PATH + lbl.name.lower() + '_clean.pkl' for lbl in label_list],
        adv_list=[DIR_PATH + lbl.name.lower() + '_clipped.pkl' for lbl in label_list],
        filename_list=[DIR_PATH + lbl.name.lower() + '_label.pkl' for lbl in label_list])
    denoising_model = AutoencoderModel(is_custom_pretrained=True).autoencoder
    return get_denoising_dataset(denoising_model, adv_dataset)


if __name__ == '__main__':
    example_calculate_adv_prediction_with_denoising()
