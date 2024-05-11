import os

from flask import Flask, render_template

app = Flask(__name__)

IMG_FOLDER = 'static/img/'
app.config["UPLOAD_FOLDER"] = IMG_FOLDER


def get_img_filename(file_templates: tuple[str, str, str]) -> tuple[str, str, str]:
    """
    The method is used to get the files name of the images.
    :param file_templates: tuple[str, str, str] of the files name patterns
    :return: tuple[str, str, str] of the files name
    """
    main_classification = None
    confusion_matrix = None
    classification_reporter = None
    paths = sorted(next(os.walk(IMG_FOLDER))[2])
    count = len(paths) - 1
    for i in range(count):
        if (main_classification is None) & (paths[count - i].startswith(file_templates[0])) & \
                (paths[count - i].endswith('.jpg')):
            main_classification = paths[count - i]
            continue
        if (confusion_matrix is None) & (paths[count - i].startswith(file_templates[1])) & \
                (paths[count - i].endswith('.jpg')):
            confusion_matrix = paths[count - i]
            continue
        if (classification_reporter is None) & (paths[count - i].startswith(file_templates[2])) & \
                (paths[count - i].endswith('.jpg')):
            classification_reporter = paths[count - i]
            continue
    return main_classification, confusion_matrix, classification_reporter


@app.route('/')
def index():
    main_classification, confusion_matrix, classification_reporter = get_img_filename(('img_main_classification_',
                                                                                       'img_confusion_matrix_',
                                                                                       'img_classification_reporter_'))
    img = {'main_classification': app.config["UPLOAD_FOLDER"] + main_classification,
           'confusion_matrix': app.config["UPLOAD_FOLDER"] + confusion_matrix,
           'classification_reporter': app.config["UPLOAD_FOLDER"] + classification_reporter}
    return render_template('index.html', img=img)
