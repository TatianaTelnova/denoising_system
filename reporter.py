import os
import pathlib
import pickle
from datetime import datetime

import torch

from label import Label
from plot import plot_main_classification, plot_confusion_matrix, plot_classification_params
from server import app


def calculate_prediction(prediction_model: torch.nn.Module, label_tensor: torch.IntTensor,
                         img_tensor: torch.Tensor) -> float:
    """
    The function calculates the prediction, find the accuracy of the model, form file with this data.
    :param prediction_model: model
    :param label_tensor: (tensor) true labels
    :param img_tensor: (tensor) input image
    :return: file name in temp_dir
    """
    temp_dir_path = pathlib.Path('./temp_prediction')
    output = prediction_model(img_tensor)
    if not temp_dir_path.exists():
        temp_dir_path.mkdir(parents=True)
    _, predicted = torch.max(output, dim=1)
    accuracy = float(torch.sum(predicted == label_tensor)) / label_tensor.shape[0]
    with open(temp_dir_path.joinpath('prediction_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pckl'), 'wb') as f:
        pickle.dump({'true_label': label_tensor, 'predicted_label': output, 'accuracy': accuracy}, f)
    return accuracy


def prepare_img_for_report() -> None:
    """
    The method prepares the image for the report.
    """
    file_main = [file_path for file_path in next(os.walk('./temp_prediction/'))[2] if file_path.endswith('.pckl')][-1]
    with open('./temp_prediction/' + file_main, 'rb') as f:
        data = pickle.load(f)
    plot_main_classification(data['accuracy'], 0)
    plot_confusion_matrix(data['true_label'], torch.max(data['predicted_label'], dim=1)[1],
                          [Label(a).name for a in set(data['true_label'].tolist())])
    plot_classification_params(data['true_label'], torch.max(data['predicted_label'], dim=1)[1],
                               [Label(a).name for a in set(data['true_label'].tolist())])


def reporter() -> None:
    """
    The method starts the report with images.
    """
    prepare_img_for_report()
    app.run(host='127.0.0.1', port=3000, debug=True)
