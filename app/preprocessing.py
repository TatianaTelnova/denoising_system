import os
from itertools import groupby

from torchvision import transforms
from typing import Final

PREPROCESSING_TRANSFORM: Final = transforms.Compose([
    transforms.Resize(200),
    transforms.ToTensor()
])


def delete_old_files(limit: int, file_dir: str):
    """
    Deletes the oldest files in the given folder if it exceeds the given limit
    :param limit: the maximum number of files
    :param file_dir: the directory path
    """
    all_files = [file_dir + file_name for file_name in next(os.walk(file_dir))[2]]
    if len(all_files) > limit:
        count = (len(all_files) - limit) // 3 + 1
        old_plt_list = [sorted(list(g))[0:count] for k, g in
                        groupby(all_files, key=lambda k: k.rpartition('_')[0].rpartition('_')[0])]
        for old_file in old_plt_list:
            for file in old_file:
                os.remove(file)


def find_files_in_dir(dir_path: str, ext: str):
    """
    Returns a list of all files in the given directory or None if no files are found
    :param dir_path: path to the directory
    :param ext: file extension
    :return: list of all files
    """
    list_of_files = [file_path for file_path in next(os.walk(dir_path))[2] if file_path.endswith(ext)]
    if len(list_of_files) == 0:
        return None
    else:
        return [dir_path + file_name for file_name in list_of_files]
