import json
import pathlib
from datetime import datetime

import torch

from app.log_config import setup_custom_logger
from app.preprocessing import delete_old_files

logger = setup_custom_logger(__name__)


def calculate_accuracy(prediction_model: torch.nn.Module,
                       true_tensor: torch.IntTensor,
                       data_tensor: torch.Tensor) -> dict:
    """
    Returns the accuracy of a prediction model
    :param prediction_model: prediction model
    :param true_tensor: (tensor) true labels
    :param data_tensor: (tensor) predicted labels
    :return: accuracy in [0,1]
    """
    output = torch.softmax(prediction_model(data_tensor), dim=1)
    _, predicted = torch.max(output, dim=1)
    return {"accuracy": float(torch.sum(predicted == true_tensor)) / true_tensor.shape[0],
            "output": output}


def calculate_prediction_with_dump(prediction_model: torch.nn.Module,
                                   label_tensor: torch.IntTensor,
                                   img_tensor: torch.Tensor) -> float:
    """
    Returns the accuracy of a prediction model with data for a single case, dumps predictions in a file
    :param prediction_model: prediction model
    :param label_tensor: (tensor) true labels
    :param img_tensor: (tensor) input image
    :return: accuracy
    """
    output = calculate_accuracy(prediction_model, label_tensor, img_tensor)
    temp_dir_path = pathlib.Path('temp_prediction')
    if not temp_dir_path.exists():
        temp_dir_path.mkdir(parents=True)
    else:
        delete_old_files(5, str(temp_dir_path))
    result_filename = temp_dir_path.joinpath('prediction_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    with open(result_filename, 'w') as f:
        json.dump({'true_label': label_tensor.tolist(),
                   'predicted_label': output['output'].tolist(),
                   'accuracy': output['accuracy']}, f)
        logger.info(f"Результат Accuracy сохранен в файл '{result_filename}'")
    return output['accuracy']


def calculate_adv_prediction_with_dump(prediction_model: torch.nn.Module,
                                       label_tensor: torch.IntTensor,
                                       img_clean_tensor: torch.Tensor,
                                       img_adv_tensor: torch.Tensor) -> tuple[float, float]:
    """
    Returns the accuracy of a prediction model with clean and adversarial image data, dumps predictions in a file
    :param prediction_model: prediction model
    :param label_tensor: (tensor) true labels
    :param img_clean_tensor: (tensor) clean image data
    :param img_adv_tensor: (tensor) adversarial image data
    :return: accuracy
    """
    clean_output = calculate_accuracy(prediction_model, label_tensor, img_clean_tensor)
    adv_output = calculate_accuracy(prediction_model, label_tensor, img_adv_tensor)
    temp_dir_path = pathlib.Path('temp_prediction')
    if not temp_dir_path.exists():
        temp_dir_path.mkdir(parents=True)
    result_filename = temp_dir_path.joinpath('prediction_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    with open(result_filename, 'w') as f:
        json.dump({'true_label': label_tensor.tolist(),
                   'predicted_true_label': clean_output['output'].tolist(),
                   'accuracy_true': clean_output['accuracy'],
                   'predicted_adv_label': adv_output['output'].tolist(),
                   'accuracy_adv': adv_output['accuracy']}, f)
        logger.info(f"Pезультат Accuracy сохранен в файл '{result_filename}'")
    return clean_output['accuracy'], adv_output['accuracy']


def calculate_adv_prediction_multi_with_dump(prediction_model: torch.nn.Module,
                                             label_tensor: torch.IntTensor,
                                             img_tensor_dict: dict[str, torch.Tensor]) -> list[float]:
    """
    Returns the accuracy of a prediction model with data for multiple cases, dumps predictions in a file
    :param prediction_model: prediction model
    :param label_tensor: (tensor) true labels
    :param img_tensor_dict: (dict) data for multiple cases
    :return: accuracy
    """
    results = {}
    for case_name, img_tensor in img_tensor_dict.items():
        output = calculate_accuracy(prediction_model, label_tensor, img_tensor)
        results['predicted_' + case_name] = output['output'].tolist()
        results['accuracy_' + case_name] = output['accuracy']
    temp_dir_path = pathlib.Path('temp_prediction')
    if not temp_dir_path.exists():
        temp_dir_path.mkdir(parents=True)
    result_filename = temp_dir_path.joinpath('prediction_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    results.update({'true_label': label_tensor.tolist()})
    with open(result_filename, 'w') as f:
        json.dump(results, f)
        logger.info(f"Результат классификации сохранен в файл '{result_filename}'")
    return [res for k, res in results.items() if k.startswith('accuracy_')]
