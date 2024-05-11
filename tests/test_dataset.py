import random

import allure
import pytest
import torch

from app import preprocessing
from app.dataset import AdvDataset, AdvImgDataset


def get_param(val):
    return "param=({0})".format(str(val))


@allure.feature('Test for AdvDataset')
class TestAdvDataset:
    BOWL_DATA = {'label': 0,
                 'clean_path': 'D:/noise/data/adv/deepfool_tensor/bowl/bowl_clean1.pckl',
                 'adv_path': 'D:/noise/data/adv/deepfool_tensor/bowl/bowl_adv.pckl',
                 'filename_path': 'D:/noise/data/adv/deepfool_tensor/bowl/bowl_label.pckl',
                 'count': 84}
    CAT_DATA = {'label': 2,
                'clean_path': 'D:/noise/data/adv/deepfool_tensor/cat/cat_clean1.pckl',
                'adv_path': 'D:/noise/data/adv/deepfool_tensor/cat/cat_adv.pckl',
                'filename_path': 'D:/noise/data/adv/deepfool_tensor/cat/cat_label.pckl',
                'count': 92}

    @pytest.fixture
    def bowl_adv_dataset(self) -> AdvDataset:
        """This method returns an AdvDataset object with the bowl data."""
        return AdvDataset(
            label_list=[self.BOWL_DATA['label']],
            clean_list=[self.BOWL_DATA['clean_path']],
            adv_list=[self.BOWL_DATA['adv_path']],
            filename_list=[self.BOWL_DATA['filename_path']])

    @pytest.fixture
    def bowl_cat_adv_dataset(self) -> AdvDataset:
        """This method returns an AdvDataset object with the bowl data."""
        return AdvDataset(
            label_list=[self.BOWL_DATA['label'], self.CAT_DATA['label']],
            clean_list=[self.BOWL_DATA['clean_path'], self.CAT_DATA['clean_path']],
            adv_list=[self.BOWL_DATA['adv_path'], self.CAT_DATA['adv_path']],
            filename_list=[self.BOWL_DATA['filename_path'], self.CAT_DATA['filename_path']])

    @allure.feature('Test for AdvDataset')
    @allure.story('Проверка методов класса AdvDataset')
    @allure.title('Test 1. Check __len__(), __get_item__()')
    @pytest.mark.parametrize('rand_idx,label,count,dataset',
                             [(random.randint(0, BOWL_DATA['count'] - 1), BOWL_DATA['label'], BOWL_DATA['count'],
                               'bowl_adv_dataset'),
                              (random.randint(0, BOWL_DATA['count'] - 1), BOWL_DATA['label'],
                               BOWL_DATA['count'] + CAT_DATA['count'], 'bowl_cat_adv_dataset'),
                              (random.randint(BOWL_DATA['count'], BOWL_DATA['count'] + CAT_DATA['count'] - 1),
                               CAT_DATA['label'], BOWL_DATA['count'] + CAT_DATA['count'], 'bowl_cat_adv_dataset')],
                             ids=get_param)
    def test_adv_dataset_class(self, rand_idx: int, label: int, count: int, dataset: str,
                               request: pytest.FixtureRequest):
        """
        This test checks if the AdvDataset length and get item returns the correct data.
        :param rand_idx: the random index of the data in the AdvDataset object to be checked
        :param label: the label of the data in the AdvDataset object
        :param count: the length of the data in the AdvDataset object
        :param dataset: the method name to get AdvDataset object
        :param request: the pytest fixture
        """
        with allure.step(f"get '{dataset}' dataset"):
            dataset_obj = request.getfixturevalue(dataset)
        with allure.step(f"check '{dataset}' length with {count} items"):
            assert dataset_obj.__len__() == count
        with allure.step(f"get '{dataset}' item with index {rand_idx}"):
            item = dataset_obj.__getitem__(rand_idx)
        with allure.step("check item data is 'tuple[int, torch.Tensor(), torch.Tensor, str]'"):
            assert (item[0] == label)
            assert (item[1].dtype == torch.float32) & (item[1].shape == (3, 200, 200))
            assert (item[2].dtype == torch.float32) & (item[2].shape == (3, 200, 200))
            assert (item[3].startswith('img')) & (item[3].endswith('.jpg'))

    @allure.feature('Test for AdvDataset')
    @allure.story('Проверка методов класса AdvDataset')
    @allure.title('Test 2. Check __get_item__() index out of range')
    @pytest.mark.parametrize('out_idx,dataset',
                             [(BOWL_DATA['count'], 'bowl_adv_dataset'),
                              (BOWL_DATA['count'] + CAT_DATA['count'], 'bowl_cat_adv_dataset')],
                             ids=get_param)
    def test_adv_dataset_out_idx(self, out_idx: int, dataset: str, request: pytest.FixtureRequest):
        """
        This test checks if the AdvDataset length returns the exception with data index out of range.
        :param out_idx: the index of the data in the AdvDataset object to be checked
        :param dataset: the method name to get AdvDataset object
        :param request: the pytest fixture
        """
        with allure.step(f"get '{dataset}' item with index {out_idx}"):
            with pytest.raises(IndexError) as e_info:
                request.getfixturevalue(dataset).__getitem__(out_idx)
            assert e_info.value.args[0] == 'Index out of range'

    @allure.feature('Test for AdvDataset')
    @allure.story('Проверка методов класса AdvDataset')
    @allure.title('Test 3. Check get_sample()')
    @pytest.mark.parametrize('data_label,dataset',
                             [([(BOWL_DATA['label'], random.randint(1, BOWL_DATA['count']))], 'bowl_adv_dataset'),
                              ([(BOWL_DATA['label'], BOWL_DATA['count']),
                                (CAT_DATA['label'], random.randint(1, CAT_DATA['count']))], 'bowl_cat_adv_dataset')],
                             ids=get_param)
    def test_adv_dataset_sample(self, data_label: list[tuple[int, int]], dataset: str, request: pytest.FixtureRequest):
        """
        This test checks if the AdvDataset sample returns the correct data.
        :param data_label: the data label of samples to be checked
        :param dataset: the method name to get AdvDataset object
        :param request: the pytest fixture
        """
        with allure.step('prepare expected label list, expected data count'):
            label_list = []
            count = 0
            for item in data_label:
                label_list += [item[0] for _ in range(item[1])]
                count += item[1]
        with allure.step(f"get '{dataset}' dataset"):
            data = request.getfixturevalue(dataset).get_sample(count)
        with allure.step(f"check sample data is 'tuple[list[int], torch.Tensor, torch.Tensor, list[str]]'"):
            assert (data[0] == label_list)
            assert (data[1].dtype == torch.float32) & (tuple(data[1].size()) == (count, 3, 200, 200))
            assert (data[2].dtype == torch.float32) & (tuple(data[2].size()) == (count, 3, 200, 200))
            assert ([(str(img_filename).startswith('img')) & (str(img_filename).endswith('.jpg')) for img_filename in
                     data[3]] == [True for _ in range(count)])

    @allure.feature('Test for AdvDataset')
    @allure.story('Проверка методов класса AdvDataset')
    @allure.title('Test 4. Check get_all()')
    @pytest.mark.parametrize('data_label,dataset',
                             [([(BOWL_DATA['label'], BOWL_DATA['count'])], 'bowl_adv_dataset'),
                              ([(BOWL_DATA['label'], BOWL_DATA['count']), (CAT_DATA['label'], CAT_DATA['count'])],
                               'bowl_cat_adv_dataset')],
                             ids=get_param)
    def test_adv_dataset_all(self, data_label: list[tuple[int, int]], dataset: str, request: pytest.FixtureRequest):
        """
        This test checks if the AdvDataset all data returns the correct data.
        :param data_label: the data label of samples to be checked
        :param dataset: the method name to get AdvDataset object
        :param request: the pytest fixture
        """
        with allure.step('prepare expected label list, expected data count'):
            label_list = []
            count = 0
            for item in data_label:
                count += item[1]
                label_list += [item[0] for _ in range(item[1])]
        with allure.step(f"get '{dataset}' dataset"):
            data = request.getfixturevalue(dataset).get_all()
        with allure.step(f"check all data is 'tuple[list[int], torch.Tensor, torch.Tensor, list[str]]'"):
            assert (data[0] == label_list)
            assert (data[1].dtype == torch.float32) & (tuple(data[1].size()) == (count, 3, 200, 200))
            assert (data[2].dtype == torch.float32) & (tuple(data[2].size()) == (count, 3, 200, 200))
            assert ([(str(img_filename).startswith('img')) & (str(img_filename).endswith('.jpg')) for img_filename in
                     data[3]] == [True for _ in range(count)])


@allure.feature('Test for AdvImgDataset')
class TestAdvImgDataset:
    BOWL_IMG_DATA = {'label': 0,
                     'bowl_img_path': 'D:/noise/data/generated/val/bowl',
                     'count': 150}
    CAT_IMG_DATA = {'label': 2,
                    'cat_img_path': 'D:/noise/data/generated/val/cat',
                    'count': 150}

    @pytest.fixture
    def bowl_adv_img_dataset(self) -> AdvImgDataset:
        """This method returns an AdvImgDataset object with the images with bowl."""
        return AdvImgDataset(
            label_list=[self.BOWL_IMG_DATA['label']],
            img_list=[self.BOWL_IMG_DATA['bowl_img_path']],
            transform=preprocessing.PREPROCESSING_TRANSFORM)

    @pytest.fixture
    def bowl_cat_adv_img_dataset(self) -> AdvImgDataset:
        """This method returns an AdvImgDataset object with the images with bowl."""
        return AdvImgDataset(
            label_list=[self.BOWL_IMG_DATA['label'], self.CAT_IMG_DATA['label']],
            img_list=[self.BOWL_IMG_DATA['bowl_img_path'], self.CAT_IMG_DATA['cat_img_path']],
            transform=preprocessing.PREPROCESSING_TRANSFORM)

    @allure.feature('Test for AdvImgDataset')
    @allure.story('Проверка методов класса AdvImgDataset')
    @allure.title('Test 1. Check __len__(), __get_item__()')
    @pytest.mark.parametrize('rand_idx,label,count,dataset',
                             [(random.randint(0, BOWL_IMG_DATA['count'] - 1), BOWL_IMG_DATA['label'],
                               BOWL_IMG_DATA['count'], 'bowl_adv_img_dataset'),
                              (random.randint(0, BOWL_IMG_DATA['count'] - 1), BOWL_IMG_DATA['label'],
                               BOWL_IMG_DATA['count'] + CAT_IMG_DATA['count'], 'bowl_cat_adv_img_dataset'),
                              (random.randint(BOWL_IMG_DATA['count'],
                                              BOWL_IMG_DATA['count'] + CAT_IMG_DATA['count'] - 1),
                               CAT_IMG_DATA['label'], BOWL_IMG_DATA['count'] + CAT_IMG_DATA['count'],
                               'bowl_cat_adv_img_dataset')], ids=get_param)
    def test_adv_dataset_class(self, rand_idx: int, label: int, count: int, dataset: str,
                               request: pytest.FixtureRequest):
        """
        This test checks if the AdvImgDataset length and get item returns the correct data.
        :param rand_idx: the random index of the data in the AdvImgDataset object to be checked
        :param label: the label of the data in the AdvImgDataset object to be checked
        :param count: the count of the data in the AdvImgDataset object
        :param dataset: the method name to get AdvImgDataset object
        :param request: the pytest fixture
        """
        with allure.step(f"get '{dataset}' dataset"):
            data = request.getfixturevalue(dataset)
        with allure.step(f"check '{dataset}' length with {count} items"):
            assert data.__len__() == count
        with allure.step(f"get '{dataset}' item with index {rand_idx}"):
            item = data.__getitem__(rand_idx)
        with allure.step("check item data is 'tuple[int, torch.Tensor, str]'"):
            assert (item[0] == label)
            assert (item[1].dtype == torch.float32) & (item[1].shape == (3, 200, 200))
            assert (item[2].startswith('img')) & (item[2].endswith('.jpg'))

    @allure.feature('Test for AdvImgDataset')
    @allure.story('Проверка методов класса AdvImgDataset')
    @allure.title('Test 2. Check __get_item__() index out of range')
    @pytest.mark.parametrize('out_idx,dataset',
                             [(BOWL_IMG_DATA['count'], 'bowl_adv_img_dataset'),
                              (BOWL_IMG_DATA['count'] + CAT_IMG_DATA['count'], 'bowl_cat_adv_img_dataset')],
                             ids=get_param)
    def test_adv_dataset_out_idx(self, out_idx: int, dataset: str, request: pytest.FixtureRequest):
        """
        This test checks if the AdvImgDataset length returns the exception with data index out of range.
        :param out_idx: the index of the data in the AdvImgDataset object to be checked
        :param dataset: the method name to get AdvImgDataset object
        """
        with allure.step(f"get '{dataset}' item with index {out_idx}"):
            with pytest.raises(IndexError) as e_info:
                request.getfixturevalue(dataset).__getitem__(out_idx)
            assert e_info.value.args[0] == 'Index out of range'

    @allure.feature('Test for AdvImgDataset')
    @allure.story('Проверка методов класса AdvImgDataset')
    @allure.title('Test 3. Check get_sample()')
    @pytest.mark.parametrize('data_label,dataset',
                             [([(BOWL_IMG_DATA['label'], random.randint(1, BOWL_IMG_DATA['count']))],
                               'bowl_adv_img_dataset'),
                              ([(BOWL_IMG_DATA['label'], BOWL_IMG_DATA['count']),
                                (CAT_IMG_DATA['label'], random.randint(1, CAT_IMG_DATA['count']))],
                               'bowl_cat_adv_img_dataset')], ids=get_param)
    def test_adv_dataset_sample(self, data_label: list[tuple[int, int]], dataset: str, request: pytest.FixtureRequest):
        """
        This test checks if the AdvImgDataset sample returns the correct data.
        :param data_label: the data label of samples to be checked
        :param dataset: the method name to get AdvImgDataset object
        :param request: the pytest fixture
        """
        with allure.step('prepare expected label tensor, expected data count'):
            label_tensor = torch.empty(0)
            count = 0
            for item in data_label:
                label_tensor = torch.cat((label_tensor, torch.tensor([item[0] for _ in range(item[1])])), dim=0)
                count += item[1]
        with allure.step(f"get '{dataset}' dataset"):
            data = request.getfixturevalue(dataset).get_sample(count)
        with allure.step("check sample data is 'tuple[list[int], torch.Tensor, torch.Tensor, list[str]]'"):
            assert (sum(data[0] == label_tensor) == count)
            assert (data[1].dtype == torch.float32) & (tuple(data[1].size()) == (count, 3, 200, 200))
            assert ([(str(img_filename).startswith('img')) & (str(img_filename).endswith('.jpg')) for img_filename in
                     data[2]] == [True for _ in range(count)])

    @allure.feature('Test for AdvImgDataset')
    @allure.story('Проверка методов класса AdvImgDataset')
    @allure.title('Test 4. Check get_all()')
    @pytest.mark.parametrize('data_label,dataset',
                             [([(BOWL_IMG_DATA['label'], BOWL_IMG_DATA['count'])], 'bowl_adv_img_dataset'),
                              ([(BOWL_IMG_DATA['label'], BOWL_IMG_DATA['count']),
                                (CAT_IMG_DATA['label'], CAT_IMG_DATA['count'])], 'bowl_cat_adv_img_dataset')],
                             ids=get_param)
    def test_adv_dataset_all(self, data_label: list[tuple[int, int]], dataset: str, request: pytest.FixtureRequest):
        """
        This test checks if the AdvImgDataset all data returns the correct data.
        :param data_label: the data label of samples to be checked
        :param dataset: the method name to get AdvImgDataset object
        :param request: the pytest fixture
        """
        with allure.step('prepare expected label tensor, expected data count'):
            label_tensor = torch.empty(0)
            count = 0
            for item in data_label:
                count += item[1]
                label_tensor = torch.cat((label_tensor, torch.tensor([item[0] for _ in range(item[1])])), dim=0)
        with allure.step(f"get '{dataset}' dataset"):
            data = request.getfixturevalue(dataset).get_all()
        with allure.step("check all data is 'tuple[list[int], torch.Tensor, torch.Tensor, list[str]]'"):
            assert (sum(data[0] == label_tensor) == count)
            assert (data[1].dtype == torch.float32) & (tuple(data[1].size()) == (count, 3, 200, 200))
            assert ([(str(img_filename).startswith('img')) & (str(img_filename).endswith('.jpg')) for img_filename in
                     data[2]] == [True for _ in range(count)])
