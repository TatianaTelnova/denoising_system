import os
import pickle

import torch
from PIL import Image
from torchvision import datasets, transforms


class AdvDataset:
    def __init__(self, label_list: list[int], clean_list: list[str], adv_list: list[str],
                 filename_list: list[str] = None):
        self.label_list = label_list
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
        if idx >= self.__get_count():
            raise IndexError('Index out of range')
        label = -1
        clean_tensor = None
        adv_tensor = None
        filename = None
        for i in range(len(self.clean_list)):
            count = self.__get_count_for_class(i)
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

    def __get_img_data(self, img_list_idx, num_samples: int) -> tuple[list[int], torch.Tensor, torch.Tensor, list[str]]:
        label = [self.label_list[img_list_idx] for _ in range(num_samples)]
        with open(self.clean_list[img_list_idx], 'rb') as f:
            clean_tensor = pickle.load(f)[:num_samples, :, :, :]
        with open(self.adv_list[img_list_idx], 'rb') as f:
            adv_tensor = pickle.load(f)[:num_samples, :, :, :]
        if self.filename_list is not None:
            with open(self.filename_list[img_list_idx], 'rb') as f:
                filename = pickle.load(f)[:num_samples]
        return label, clean_tensor, adv_tensor, filename

    def __get_img_data_all(self, img_list_idx) -> tuple[list[int], torch.Tensor, torch.Tensor, list[str]]:
        with open(self.clean_list[img_list_idx], 'rb') as f:
            clean_tensor = pickle.load(f)
            label = [self.label_list[img_list_idx] for _ in range(clean_tensor.shape[0])]
        with open(self.adv_list[img_list_idx], 'rb') as f:
            adv_tensor = pickle.load(f)
        if self.filename_list is not None:
            with open(self.filename_list[img_list_idx], 'rb') as f:
                filename = pickle.load(f)
        else:
            filename = ['-' for _ in range(clean_tensor.shape[0])]
        return label, clean_tensor, adv_tensor, filename

    def get_all(self) -> tuple[list[int], torch.Tensor, torch.Tensor, list[str]]:
        clean_tensor = torch.empty(0)
        adv_tensor = torch.empty(0)
        filename = []
        label = []
        for i in range(len(self.clean_list)):
            label_item, clean_tensor_item, adv_tensor_item, filename_item = self.__get_img_data_all(i)
            label += label_item
            clean_tensor = torch.cat((clean_tensor_item, clean_tensor), dim=0)
            adv_tensor = torch.cat((adv_tensor_item, adv_tensor), dim=0)
            filename += filename_item
        return label, clean_tensor, adv_tensor, filename

    def get_sample(self, num_samples) -> tuple[list[int], torch.Tensor, torch.Tensor, list[str]]:
        clean_tensor = torch.empty(0)
        adv_tensor = torch.empty(0)
        filename = []
        label = []
        for i in range(len(self.clean_list)):
            count = self.__get_count_for_class(i)
            if num_samples > count:
                num_samples -= count
                label_item, clean_tensor_item, adv_tensor_item, filename_item = self.__get_img_data_all(i)
                label += label_item
                clean_tensor = torch.cat((clean_tensor_item, clean_tensor), dim=0)
                adv_tensor = torch.cat((adv_tensor_item, adv_tensor), dim=0)
                filename += filename_item
            else:
                label_item, clean_tensor_item, adv_tensor_item, filename_item = self.__get_img_data(i, num_samples)
                label += label_item
                clean_tensor = torch.cat((clean_tensor_item, clean_tensor), dim=0)
                adv_tensor = torch.cat((adv_tensor_item, adv_tensor), dim=0)
                filename += filename_item
                break
        return label, clean_tensor, adv_tensor, filename


class AdvImgDataset:
    __DEFAULT_TRANSFORM = transforms.Compose(transforms=transforms.ToTensor())

    def __init__(self, label_list: list[int], img_list: list[str], transform: transforms = None):
        self.label_list = label_list
        self.img_list = img_list
        self.transform = transform

    def __get_count(self) -> int:
        count = 0
        for img_file in self.img_list:
            count += len(next(os.walk(img_file))[2])
        return count

    def __get_count_for_class(self, class_idx) -> int:
        return len(next(os.walk(self.img_list[class_idx]))[2])

    def __transform_to_tensor(self, img_tensor) -> torch.Tensor:
        if self.transform is not None:
            return self.transform(img_tensor)
        else:
            return self.__DEFAULT_TRANSFORM(img_tensor)

    def __len__(self) -> int:
        return self.__get_count()

    def __getitem__(self, idx) -> tuple[int, torch.Tensor, str]:
        if idx >= self.__get_count():
            raise IndexError('Index out of range')
        label = -1
        img_tensor = None
        filename = None
        for i in range(len(self.img_list)):
            count = self.__get_count_for_class(i)
            if idx < count:
                label = self.label_list[i]
                filename = next(os.walk(self.img_list[i]))[2][idx]
                img_tensor = self.__transform_to_tensor(
                    Image.open(self.img_list[i] + '/' + filename))
                break
            else:
                idx -= count
        return label, img_tensor, filename

    def __get_img_data(self, img_list_idx, num_samples: int = None) -> tuple[list[int], torch.Tensor, list[str]]:
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
            label_item, img_tensor_item, filename_item = self.__get_img_data(i, None)
            label += label_item
            img_tensor = torch.cat((img_tensor, img_tensor_item), dim=0)
            filename += filename_item
        return torch.IntTensor(label), img_tensor, filename

    def get_sample(self, num_samples) -> tuple[torch.IntTensor, torch.Tensor, list[str]]:
        label = []
        img_tensor = torch.empty(0)
        filename = []
        for i in range(len(self.img_list)):
            files_list = next(os.walk(self.img_list[i]))[2]
            if num_samples > len(files_list):
                num_samples -= len(files_list)
                label_item, img_tensor_item, filename_item = self.__get_img_data(i, None)
                label += label_item
                img_tensor = torch.cat((img_tensor, img_tensor_item), dim=0)
                filename += filename_item
            else:
                label_item, img_tensor_item, filename_item = self.__get_img_data(i, num_samples)
                label += label_item
                img_tensor = torch.cat((img_tensor, img_tensor_item), dim=0)
                filename += filename_item
                break
        return torch.IntTensor(label), img_tensor, filename


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
