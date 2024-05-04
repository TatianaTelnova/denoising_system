import os
import torch

from dataset import AdvDataset
from img_client import img_list_to_tensor
from model_classification import ClassificationModel


def get_predict_accuracy(model, img_input: torch.Tensor, label: torch.Tensor):
    output = model(img_input)
    _, predict = torch.max(output, dim=1)
    return float(torch.sum(predict == label) / label.size(dim=0))


def check_noise_immunity(model, adv_dataset: list[tuple]):
    img_clean_input = img_list_to_tensor([item[0] for item in adv_dataset])
    img_adv_input = img_list_to_tensor([item[1] for item in adv_dataset])
    img_label = torch.IntTensor([item[2] for item in adv_dataset])
    print(get_predict_accuracy(model, img_clean_input, img_label),
          get_predict_accuracy(model, img_adv_input, img_label))


def get_data_file_name(root_directory, class_name, pattern_filename):
    files = [all_files for all_files in next(os.walk(root_directory + class_name))[2] if
             all_files.endswith(pattern_filename)]
    assert len(files) == 1
    return root_directory + class_name + '/' + files[0]


if __name__ == '__main__':
    root_directory = 'D:/noise/data/adv/deepfool_tensor/'
    clean_file = '_clean1.pckl'
    adv_file = '_adv.pckl'
    label_file = '_label.pckl'

    class_names = next(os.walk(root_directory))[1]
    clean_list = []
    adv_list = []
    filename_list = []
    for img_class in class_names:
        clean_list.append(get_data_file_name(root_directory, img_class, clean_file))
        adv_list.append(get_data_file_name(root_directory, img_class, adv_file))
        filename_list.append(get_data_file_name(root_directory, img_class, label_file))
    dataset_adv = {
        'train': AdvDataset(
            clean_list=clean_list,
            adv_list=adv_list,
            filename_list=filename_list
        )
    }
    label, clean_tensor, adv_tensor, _ = dataset_adv.get('train').get_all()
    model = ClassificationModel(is_custom_pretrained=True).model
    print(get_predict_accuracy(model, clean_tensor, torch.IntTensor(label)),
          get_predict_accuracy(model, adv_tensor, torch.IntTensor(label)))
