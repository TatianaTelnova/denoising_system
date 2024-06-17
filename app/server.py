import os
from itertools import groupby

from flask import Flask, render_template

app = Flask(__name__)

IMG_FOLDER = 'static/img/'
app.config["UPLOAD_FOLDER"] = IMG_FOLDER


def get_img_filename(file_templates: tuple[str, str, str]) -> tuple[str, str, str]:
    """
    The method is used to get the last files name of the images.
    :param file_templates: tuple[str, str, str] of the files name patterns
    :return: tuple[str, str, str] of the files name
    """
    file_name = {file_templates[0]: None, file_templates[1]: None, file_templates[2]: None}
    paths = next(os.walk(IMG_FOLDER))[2]
    grouped_files = [sorted(list(g))[-1] for k, g in
                     groupby(paths, key=lambda k: k.rpartition('_')[0].rpartition('_')[0])]
    for template in file_templates:
        files_filter = [file_name for file_name in grouped_files if file_name.startswith(template)]
        if len(files_filter) != 0:
            file_name[template] = files_filter[0]
    return file_name[file_templates[0]], file_name[file_templates[1]], file_name[file_templates[2]]


@app.route('/')
def index():
    main_classification, confusion_matrix, classification_reporter = get_img_filename(('img_main_classification_',
                                                                                       'img_confusion_matrix_',
                                                                                       'img_classification_reporter_'))
    img = {'main_classification': app.config["UPLOAD_FOLDER"] + main_classification,
           'confusion_matrix': app.config["UPLOAD_FOLDER"] + confusion_matrix,
           'classification_reporter': app.config["UPLOAD_FOLDER"] + classification_reporter}
    return render_template('index.html', img=img)
