import pathlib

from PIL import Image


def save_img(img_filename: str, local_dir='D:/local_dir/'):
    path_local_dir = pathlib.Path(local_dir)
    if not path_local_dir.exists():
        path_local_dir.mkdir(parents=True)


# def get_img(img_id):
#
#
# def display_img(img_id):
