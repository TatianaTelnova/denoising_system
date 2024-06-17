import os
import pickle
from abc import ABCMeta, abstractmethod

import torch
from PIL import Image
from torchvision import transforms


class Dataset:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_count_for_class(self, class_idx) -> int:
        """Returns the number of samples in the class"""

    @abstractmethod
    def get_img_data(self, class_idx) -> torch.Tensor:
        """Returns the image data of the class"""

    @abstractmethod
    def get_all(self) -> tuple:
        """Returns all information about the all samples from the dataset"""

    @abstractmethod
    def get_samples(self, num_samples) -> tuple:
        """Returns all information about the samples from the dataset"""


class AdvDataset(Dataset):
    """Dataset class for the adversarial Tensor data"""

    def __init__(self, label_list: list, clean_list: list[str], adv_list: list[str],
                 filename_list: list[str] = None):
        self.label_list = label_list
        self.clean_list = clean_list
        self.adv_list = adv_list
        self.filename_list = filename_list

    def __len__(self) -> int:
        count = 0
        for clean_file in self.clean_list:
            with open(clean_file, 'rb') as f:
                count += pickle.load(f).shape[0]
        return count

    def __getitem__(self, idx) -> tuple[int, torch.Tensor, torch.Tensor, str]:
        if idx >= self.__len__():
            raise IndexError('Index out of range')
        label = -1
        clean_tensor = None
        adv_tensor = None
        filename = None
        for i in range(len(self.clean_list)):
            count = self.get_count_for_class(self.label_list[i])
            if idx < count:
                label = self.label_list[i]
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

    def get_count_for_class(self, class_idx) -> int:
        """
        Returns the number of samples in the class by class index
        :param class_idx: class index
        :return: number of samples
        """
        with open(self.clean_list[self.label_list.index(class_idx)], 'rb') as f:
            count = len(pickle.load(f))
        return count

    def get_img_data(self, img_list_idx, num_samples: int = None) -> tuple[list[int],
                                                                           torch.Tensor, torch.Tensor, list[str]]:
        if num_samples is None:
            num_samples = self.get_count_for_class(self.label_list[img_list_idx])
        with open(self.clean_list[img_list_idx], 'rb') as f:
            clean_tensor = pickle.load(f)[:num_samples]
            label = [self.label_list[img_list_idx] for _ in range(num_samples)]
        with open(self.adv_list[img_list_idx], 'rb') as f:
            adv_tensor = pickle.load(f)[:num_samples]
        if self.filename_list is not None:
            with open(self.filename_list[img_list_idx], 'rb') as f:
                filename = pickle.load(f)[:num_samples]
        else:
            filename = ['-' for _ in range(clean_tensor.shape[0])]
        return label, clean_tensor, adv_tensor, filename

    def get_all(self) -> tuple[list[int], torch.Tensor, torch.Tensor, list[str]]:
        clean_tensor = torch.empty(0)
        adv_tensor = torch.empty(0)
        filename = []
        label = []
        for i in range(len(self.clean_list)):
            label_item, clean_tensor_item, adv_tensor_item, filename_item = self.get_img_data(i, None)
            label += label_item
            clean_tensor = torch.cat((clean_tensor, clean_tensor_item), dim=0)
            adv_tensor = torch.cat((adv_tensor, adv_tensor_item), dim=0)
            filename += filename_item
        return label, clean_tensor, adv_tensor, filename

    def get_samples(self, num_samples) -> tuple[list[int], torch.Tensor, torch.Tensor, list[str]]:
        clean_tensor = torch.empty(0)
        adv_tensor = torch.empty(0)
        filename = []
        label = []
        for i in range(len(self.clean_list)):
            count = self.get_count_for_class(self.label_list[i])
            if num_samples > count:
                num_samples -= count
                label_item, clean_tensor_item, adv_tensor_item, filename_item = self.get_img_data(i, None)
                label += label_item
                clean_tensor = torch.cat((clean_tensor, clean_tensor_item), dim=0)
                adv_tensor = torch.cat((adv_tensor, adv_tensor_item), dim=0)
                filename += filename_item
            else:
                label_item, clean_tensor_item, adv_tensor_item, filename_item = self.get_img_data(i, num_samples)
                label += label_item
                clean_tensor = torch.cat((clean_tensor, clean_tensor_item), dim=0)
                adv_tensor = torch.cat((adv_tensor, adv_tensor_item), dim=0)
                filename += filename_item
                break
        return label, clean_tensor, adv_tensor, filename


class AdvImgDataset(Dataset):
    """Dataset for images with/without perturbation"""
    __DEFAULT_TRANSFORM = transforms.Compose(transforms=transforms.ToTensor())

    def __init__(self, label_list: list, img_list: list[str], transform: transforms = None):
        self.label_list = label_list
        self.img_list = img_list
        self.transform = transform

    def get_count_for_class(self, class_idx) -> int:
        return len(next(os.walk(self.img_list[self.label_list.index(class_idx)]))[2])

    def __transform_to_tensor(self, img_tensor) -> torch.Tensor:
        if self.transform is not None:
            return self.transform(img_tensor)
        else:
            return self.__DEFAULT_TRANSFORM(img_tensor)

    def __len__(self) -> int:
        count = 0
        for img_file in self.img_list:
            count += len(next(os.walk(img_file))[2])
        return count

    def __getitem__(self, idx) -> tuple[int, torch.Tensor, str]:
        if idx >= self.__len__():
            raise IndexError('Index out of range')
        label = -1
        img_tensor = None
        filename = None
        for i in range(len(self.img_list)):
            count = self.get_count_for_class(self.label_list[i])
            if idx < count:
                label = self.label_list[i]
                filename = next(os.walk(self.img_list[i]))[2][idx]
                img_tensor = self.__transform_to_tensor(
                    Image.open(self.img_list[i] + '/' + filename))
                break
            else:
                idx -= count
        return label, img_tensor, filename

    def get_img_data(self, img_list_idx, num_samples: int = None) -> tuple[list[int], torch.Tensor, list[str]]:
        files = next(os.walk(self.img_list[img_list_idx]))[2]
        if num_samples is None:
            num_samples = len(files)
        img_tensor = torch.empty(0)
        label = [self.label_list[img_list_idx] for _ in range(num_samples)]
        for i in range(num_samples):
            img = self.__transform_to_tensor(Image.open(self.img_list[img_list_idx] + '/' + files[i]))
            img_tensor = torch.cat((img_tensor, img.unsqueeze(0)), dim=0)
        return label, img_tensor, files[:num_samples]

    def get_all(self) -> tuple[torch.IntTensor, torch.Tensor, list[str]]:
        label = []
        img_tensor = torch.empty(0)
        filename = []
        for i in range(len(self.img_list)):
            label_item, img_tensor_item, filename_item = self.get_img_data(i, None)
            label += label_item
            img_tensor = torch.cat((img_tensor, img_tensor_item), dim=0)
            filename += filename_item
        return torch.IntTensor(label), img_tensor, filename

    def get_samples(self, num_samples) -> tuple[torch.IntTensor, torch.Tensor, list[str]]:
        label = []
        img_tensor = torch.empty(0)
        filename = []
        for i in range(len(self.img_list)):
            files_list = next(os.walk(self.img_list[i]))[2]
            if num_samples > len(files_list):
                num_samples -= len(files_list)
                label_item, img_tensor_item, filename_item = self.get_img_data(i, None)
                label += label_item
                img_tensor = torch.cat((img_tensor, img_tensor_item), dim=0)
                filename += filename_item
            else:
                label_item, img_tensor_item, filename_item = self.get_img_data(i, num_samples)
                label += label_item
                img_tensor = torch.cat((img_tensor, img_tensor_item), dim=0)
                filename += filename_item
                break
        return torch.IntTensor(label), img_tensor, filename
