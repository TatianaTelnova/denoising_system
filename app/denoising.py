import os
import pickle
from log_config import setup_custom_logger

import torch
from foolbox import Attack, models

from app.dataset import AdvImgDataset, AdvDataset

logger = setup_custom_logger(__name__)


def get_noise_dataset(model: torch.nn.Module, attack: Attack, dataset: AdvImgDataset, epsilon=1):
    """
    Returns object AdvDataset with noise examples
    :param model: prediction model
    :param attack: attack
    :param dataset: clean image dataset
    :param epsilon: parameter of noise size
    :return: AdvDataset with noise examples
    """
    fool_model = models.PyTorchModel(model, bounds=(0, 1))
    data = dataset.get_all()
    _, clipped, is_adv = attack(fool_model, data[1], data[0].long(), epsilons=epsilon)
    previous_count = 0
    for label, img_path in zip(dataset.label_list, dataset.img_list):
        current_count = previous_count + dataset.get_count_for_class(label)
        clipped_data = clipped[previous_count:current_count]
        is_adv_data = is_adv[previous_count:current_count]
        logger.info(f"Для класса {img_path.rpartition('/')[-1]}:")
        with open(f'{img_path}_clean.pkl', 'wb') as f:
            pickle.dump(data[1][previous_count:current_count], f)
            logger.info(f"класс '{img_path.rpartition('/')[-1]}': оригинальные изображения в '{img_path}_clean.pkl'")
        with open(f'{img_path}_clipped.pkl', 'wb') as f:
            pickle.dump(clipped_data, f)
            logger.info(f"класс '{img_path.rpartition('/')[-1]}': изображения с шумом '{type(attack).__name__}' в "
                        f"'{img_path}_clipped.pkl'")
        with open(f'{img_path}_is_adv.pkl', 'wb') as f:
            pickle.dump(is_adv_data, f)
            logger.info(f"класс '{img_path.rpartition('/')[-1]}': инфо 'является негативным объектом' в "
                        f"'{img_path}_is_adv.pkl'")
        with open(f'{img_path}_label.pkl', 'wb') as f:
            pickle.dump([img_path + '/' + file_name for file_name in next(os.walk(img_path))[2]], f)
            logger.info(f"класс '{img_path.rpartition('/')[-1]}': полное название файла в '{img_path}_label.pkl'")
        previous_count = current_count
    return AdvDataset(
        label_list=dataset.label_list,
        clean_list=[f'{img_path}_clean.pkl' for img_path in dataset.img_list],
        adv_list=[f'{img_path}_clipped.pkl' for img_path in dataset.img_list],
        filename_list=[f'{img_path}_label.pkl' for img_path in dataset.img_list]
    )


def get_denoising_dataset(model: torch.nn.Module, dataset: AdvDataset):
    """
    Returns object AdvDataset with denoising examples
    :param model: desoining model
    :param dataset: AdvDataset with noise examples
    :return: AdvDataset with denoising examples
    """
    all_adv_data = dataset.get_all()
    denoising_data = model(all_adv_data[2])
    previous_count = 0
    for label, img_path in zip(dataset.label_list, dataset.clean_list):
        current_count = previous_count + dataset.get_count_for_class(label)
        img_file_name = img_path.rpartition('_')[0]
        with open(f'{img_file_name}_denoising.pkl', 'wb') as f:
            pickle.dump(denoising_data[previous_count:current_count], f)
            logger.info(f"Класс '{img_file_name.rpartition('/')[-1]}': изображения после шумоподавления в "
                        f"'{img_file_name}_denoising.pkl'")
        previous_count = current_count
    return AdvDataset(
        label_list=dataset.label_list,
        clean_list=dataset.clean_list,
        adv_list=[f'{img_path.rpartition("_")[0]}_denoising.pkl' for img_path in dataset.clean_list],
        filename_list=dataset.filename_list)
