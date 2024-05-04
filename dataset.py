import os
import pickle

import torch
from torchvision import datasets


class AdvDataset:
    def __init__(self, clean_list: list[str], adv_list: list[str], filename_list: list[str] = None):
        self.clean_list = clean_list
        self.adv_list = adv_list
        self.filename_list = filename_list

    def __get_count(self) -> int:
        count = 0
        for clean_file in self.clean_list:
            with open(clean_file, 'rb') as f:
                count += pickle.load(f).shape[0]
        return count

    def __get_count_for_class(self, class_idx) -> int:
        with open(self.clean_list[class_idx], 'rb') as f:
            count = len(pickle.load(f))
        return count

    def __len__(self) -> int:
        return self.__get_count()

    def __getitem__(self, idx) -> tuple[int, torch.Tensor, torch.Tensor, str]:
        label = -1
        clean_tensor = None
        adv_tensor = None
        filename = None
        for i in range(len(self.clean_list)):
            count = self.__get_count_for_class(i)
            if idx < count:
                label = i
                with open(self.clean_list[i], 'rb') as f:
                    clean_tensor = pickle.load(f)[idx, :, :, :]
                with open(self.adv_list[i], 'rb') as f:
                    adv_tensor = pickle.load(f)[idx, :, :, :]
                if self.filename_list is not None:
                    with open(self.filename_list[i], 'rb') as f:
                        filename = pickle.load(f)[idx]
                break
            else:
                idx -= count
        return label, clean_tensor, adv_tensor, filename

    def get_all(self) -> tuple[list[int], torch.Tensor, torch.Tensor, list[str]]:
        clean_tensor = torch.empty(0)
        adv_tensor = torch.empty(0)
        filename = []
        label = []
        for i in range(len(self.clean_list)):
            with open(self.clean_list[i], 'rb') as f:
                temp_tensor = pickle.load(f)
                clean_tensor = torch.cat((clean_tensor, temp_tensor), dim=0)
                label += [i for j in range(temp_tensor.shape[0])]
            with open(self.adv_list[i], 'rb') as f:
                adv_tensor = torch.cat((adv_tensor, pickle.load(f)), dim=0)
            if self.filename_list is not None:
                with open(self.filename_list[i], 'rb') as f:
                    filename += pickle.load(f)
        return label, clean_tensor, adv_tensor, filename

    # todo
    # def get_sample(self, num_samples):


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
    print(dataset_adv.get('train').get_all())


class Adv2Dataset:
    """
    This class is used for the adversarial dataset.
    """

    def __init__(self, root_clean, root_adv, transform=None):
        """
        This method is used to initialize the class with paths to the clean images and the adversarial images.
        :param root_clean: the path to the clean images
        :param root_adv: the path to the adversarial images
        :param transform: the transform to be applied to the images
        """
        self.data_clean = datasets.ImageFolder(root_clean, transform=transform)
        self.data_adv = datasets.ImageFolder(root_adv, transform=transform)

    def __len__(self):
        """
        This method is used to return the length of the dataset.
        :return: clean images dataset length, length of the adversarial images dataset is the same
        """
        return len(self.data_clean.imgs)

    def __getitem__(self, idx):
        """
        This method is used to return the tensor of the image and the label at the given index.
        :param idx: the index of the image
        :return: the tuple (img_clean_tensor, img_adv_tensor, label)
        """
        img_clean, _ = self.data_clean[idx]
        img_adv, label = self.data_adv[idx]
        return img_clean, img_adv, label

    def get_samples(self, num_samples):
        """
        This method is used to return a number of samples from the dataset.
        :param num_samples: the number of samples from 0 to num_samples-1 to return
        :return: the list of tuple (img_clean_path, img_adv_path, label)
        """
        # data_sample = []
        # for data_sample_clean, data_sample_adv in zip(self.data_clean.imgs[:num_samples],
        #                                               self.data_adv.imgs[:num_samples]):
        #     data_sample.append((data_sample_clean[0], data_sample_adv[0], data_sample_adv[1]))
        # return data_sample

        return [(data_sample_clean[0], data_sample_adv[0], data_sample_adv[1]) for data_sample_clean, data_sample_adv in
                zip(self.data_clean.imgs[:num_samples],
                    self.data_adv.imgs[:num_samples])]

    def get_img(self, idx):
        data_clean = self.data_clean.imgs[idx]
        data_adv = self.data_adv.imgs[idx]
        return data_clean[0], data_adv[0], data_adv[1]

    def get_samples_tensor(self, idx):
        return [self.__getitem__(i) for i in range(idx)]
