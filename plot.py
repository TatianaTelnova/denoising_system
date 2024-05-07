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
    # plt.show()
    return saved_filename


def plot_confusion_matrix(y_true, y_predicted, labels) -> str:
    """
    The function plots the confusion matrix results, save the image and return the filename.
    :param y_true: true labels
    :param y_predicted: predicted labels
    :param labels: labels name
    :return: filename of saved image
    """
    result_matrix = confusion_matrix(y_true, y_predicted)
    fig, ax = plt.subplots()
    ax.imshow(result_matrix)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=40)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.xlabel('Предсказанные', fontweight='bold')
    plt.ylabel('Реальные', rotation=90, fontweight='bold')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(result_matrix[i, j]), ha="center", va="center")
    saved_filename = ROOT_IMG_FOLDER + 'img_confusion_matrix_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
    plt.savefig(saved_filename, dpi=300, bbox_inches='tight')
    # plt.show()
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
    saved_filename = ROOT_IMG_FOLDER + 'img_classification_reporter_' + datetime.now().strftime('%Y%m%d_%H%M%S') \
                     + '.jpg'
    plt.savefig(saved_filename, dpi=300, bbox_inches='tight')
    # plt.show()
    return saved_filename
