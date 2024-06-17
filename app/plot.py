import enum

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT_IMG_FOLDER = 'static/img/'


def plot_main_classification(clean_result: float, adv_result: float) -> str:
    """
    The function plots the main classification results, save the image and return the filename.
    :param clean_result: result for images without noise
    :param adv_result: result for images with noise
    :return: filename of saved image
    """
    x = ['Исходные изображения', 'Зашумленные изображения']
    y = np.array([clean_result, adv_result])
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set(ylabel='Доля правильных ответов (Accuracy)')
    ax.set_ylim(0, 1)
    saved_filename = ROOT_IMG_FOLDER + 'img_main_classification_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
    fig.savefig(saved_filename, dpi=300, bbox_inches='tight')
    return saved_filename


def plot_main_classification_multi(result: dict[str, float]) -> str:
    x = []
    y = []
    for case_name, result in result.items():
        x.append(case_name)
        y.append(result)
    y = np.array(y)
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set(ylabel='Доля правильных ответов (Accuracy)')
    ax.set_ylim(0, 1)
    saved_filename = ROOT_IMG_FOLDER + 'img_main_classification_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
    fig.savefig(saved_filename, dpi=300, bbox_inches='tight')
    # plt.show()
    return saved_filename


def plot_confusion_matrix(y_true: list[int], y_predicted: list[int], labels: list) -> str:
    """
    The function plots the confusion matrix results, save the image and return the filename.
    :param y_true: true labels
    :param y_predicted: predicted labels
    :param labels: all labels name, only useful has been printed on the image
    :return: filename of saved image
    """
    result_matrix = confusion_matrix(y_true, y_predicted)
    labels = [labels[filtered] for filtered in range(result_matrix.shape[0])]
    fig, ax = plt.subplots()
    ax.imshow(result_matrix, cmap='YlOrBr')
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=40)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.xlabel('Предсказанные', fontweight='bold')
    plt.ylabel('Реальные', rotation=90, fontweight='bold')
    for i in range(result_matrix.shape[0]):
        for j in range(result_matrix.shape[1]):
            ax.text(j, i, int(result_matrix[i, j]), ha="center", va="center")
    saved_filename = ROOT_IMG_FOLDER + 'img_confusion_matrix_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
    plt.savefig(saved_filename, dpi=300, bbox_inches='tight')
    # plt.show()
    return saved_filename


def plot_confusion_matrix_multi(y_true: list[int], y_predicted: dict[str, list[int]], labels: enum) -> str:
    data = {}
    for case_name, pred in y_predicted.items():
        data[case_name] = confusion_matrix(y_true, pred)
        data[case_name + '_lbl_names'] = [labels(label_conf).name for label_conf in set(pred + y_true)]
    fig, ax = plt.subplots(nrows=len(data) // 2, ncols=1)
    i = 0
    for case_name, data_values in data.items():
        if not case_name.endswith('_lbl_names'):
            ax[i].imshow(data_values, cmap='YlOrBr')
            ax[i].set_title(str(case_name).capitalize(), fontweight='bold')
            ax[i].set_xticks(np.arange(len(data[case_name + '_lbl_names'])), labels=data[case_name + '_lbl_names'],
                             rotation=40)
            ax[i].set_yticks(np.arange(len(data[case_name + '_lbl_names'])), labels=data[case_name + '_lbl_names'])
            ax[i].set_xlabel('Предсказанные', fontweight='bold')
            ax[i].set_ylabel('Реальные', rotation=90, fontweight='bold')
            for j in range(data_values.shape[0]):
                for k in range(data_values.shape[1]):
                    ax[i].text(k, j, int(data_values[j][k]), ha="center", va="center")
            i += 1
    saved_filename = ROOT_IMG_FOLDER + 'img_confusion_matrix_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
    plt.tight_layout()
    plt.savefig(saved_filename, dpi=300, bbox_inches='tight')
    return saved_filename


def plot_classification_params(y_true: list[int], y_predicted: list[int], labels: list):
    """
    The function plots the classification reporter results, save the image and return the filename.
    :param y_true: true labels
    :param y_predicted: predicted labels
    :param labels: labels name
    :return: filename of saved image
    """
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_predicted)
    indexes = [idx for idx in range(len(support)) if support[idx] != 0]
    precision_filter, recall_filter, f_score_filter = [], [], []
    for idx in indexes:
        precision_filter.append(precision[idx])
        recall_filter.append(recall[idx])
        f_score_filter.append(f_score[idx])
    data = np.transpose([precision_filter, recall_filter, recall_filter])
    fig, ax = plt.subplots()
    ax.pcolor(data)
    xlabs = ['Precision', 'Recall', 'F-score']
    ax.set_xticks(np.arange(len(xlabs)) + 0.5, labels=xlabs, fontweight='bold')
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels=labels)
    fig = plt.gcf()
    fig.set_size_inches(10, 3)
    for i in range(len(xlabs)):
        for j in range(len(labels)):
            ax.text(i + 0.5, j + 0.5, round(data[j][i], 2), ha="center", va="center")
    saved_filename = ROOT_IMG_FOLDER + 'img_classification_reporter_' + datetime.now().strftime(
        '%Y%m%d_%H%M%S') + '.jpg'
    plt.savefig(saved_filename, dpi=300, bbox_inches='tight')
    # plt.show()
    return saved_filename


def plot_classification_params_multi(y_true: list[int], y_predicted: dict[str, list[int]], labels: enum):
    data = {}
    for case_name, pred in y_predicted.items():
        precision, recall, f_score, support = precision_recall_fscore_support(y_true, pred)
        indexes = [idx for idx in range(len(support)) if support[idx] != 0]
        precision_filter, recall_filter, f_score_filter = [], [], []
        for idx in indexes:
            precision_filter.append(precision[idx])
            recall_filter.append(recall[idx])
            f_score_filter.append(f_score[idx])
        data[case_name] = np.transpose([precision_filter, recall_filter, f_score_filter])
    fig, ax = plt.subplots(nrows=len(data), ncols=1)
    i = 0
    for case_name, data_values in data.items():
        ax[i].pcolor(data_values)
        xlabs = ['Precision', 'Recall', 'F-score']
        ax[i].set_title(str(case_name).capitalize(), fontweight='bold')
        ax[i].set_xticks(np.arange(len(xlabs)) + 0.5, labels=xlabs, fontweight='bold')
        ax[i].set_yticks(np.arange(len(set(y_true))) + 0.5, labels=[labels(f).name for f in set(y_true)])
        fig = plt.gcf()
        fig.set_size_inches(10, 3)
        for j in range(len(xlabs)):
            for k in range(len(set(y_true))):
                ax[i].text(j + 0.5, k + 0.5, round(data_values[k][j], 2), ha="center", va="center")
        i += 1
    plt.tight_layout()
    saved_filename = ROOT_IMG_FOLDER + 'img_classification_reporter_' + datetime.now().strftime(
        '%Y%m%d_%H%M%S') + '.jpg'
    plt.savefig(saved_filename, dpi=300, bbox_inches='tight')
    return saved_filename
