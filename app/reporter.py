import json
import os
from itertools import groupby

import torch

from app.log_config import setup_custom_logger
from app.preprocessing import delete_old_files, find_files_in_dir
from label import Label
from plot import ROOT_IMG_FOLDER, plot_main_classification_multi, plot_confusion_matrix_multi, \
    plot_classification_params_multi
from server import app

logger = setup_custom_logger(__name__)

LIMIT = 15


def is_new_prediction_exists(last_prediction_file: str) -> bool:
    """
    Returns True if the last prediction file is newer than the last plot result
    :param last_prediction_file: path to last prediction file
    :return: True or False
    """
    files = find_files_in_dir(ROOT_IMG_FOLDER, 'jpg')
    if files is not None:
        return os.path.getmtime(last_prediction_file) > os.path.getmtime(files[-1])
    return True


def prepare_img_for_report() -> tuple[str, str, str]:
    """
    The method prepares the image for the report.
    """
    file_main = 'temp_prediction/' + [file_path for file_path in next(os.walk('temp_prediction/'))[2]
                                      if file_path.endswith('.json')][-1]
    if is_new_prediction_exists(file_main):
        if len(file_main) == 0:
            logger.warn(f"Files with classification result in dir = 'temp_prediction' not found!")
            return '', '', ''
        with open(file_main, 'r') as f:
            data = json.load(f)
        delete_old_files(LIMIT, ROOT_IMG_FOLDER)
        main_classification = plot_main_classification_multi(
            {case_name.rpartition('_')[-1]: item for case_name, item in data.items()
             if case_name.startswith('accuracy')})
        confusion_matrix = plot_confusion_matrix_multi(
            data['true_label'], {case_name.rpartition('_')[-1]: torch.max(torch.Tensor(item), dim=1)[1].tolist() for
                                 case_name, item in data.items() if case_name.startswith('predicted')}, Label)
        classification_params = plot_classification_params_multi(
            data['true_label'], {case_name.rpartition('_')[-1]: torch.max(torch.Tensor(item), dim=1)[1].tolist() for
                                 case_name, item in data.items() if case_name.startswith('predicted')}, Label)
        logger.info(f"результаты сохранены в файлы: '{main_classification}', '{confusion_matrix}', "
                    f"'{classification_params}'")
        return main_classification, confusion_matrix, classification_params
    else:
        all_plt_list = [file_path for file_path in next(os.walk(ROOT_IMG_FOLDER))[2] if file_path.endswith('.jpg')]
        last_plt_list = [sorted(list(g))[-1] for k, g in
                         groupby(all_plt_list, key=lambda k: k.rpartition('_')[0].rpartition('_')[0])]
        logger.info(f"результаты сохранены в файлы: '{last_plt_list[0]}', '{last_plt_list[1]}', '{last_plt_list[2]}'")
        return last_plt_list[0], last_plt_list[1], last_plt_list[2]


def reporter() -> None:
    """
    The method starts the report with images.
    """
    prepare_img_for_report()
    app.run(host='127.0.0.1', port=3000, debug=False)
