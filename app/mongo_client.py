from pymongo import MongoClient


class MongoDBConnectionManager:
    """
    This class is used to connect to the database.
    """

    def __enter__(self):
        """
        This function is used to connect to the database localhost with default port.
        :return: MongoClient object
        """
        self.connection = MongoClient('localhost', 27017)
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        This function is used to close the connection to the database.
        :param exc_type: type of exception
        :param exc_val: value of exception
        :param exc_tb: traceback of exception
        :return: MongoClient object closed
        """
        self.connection.close()


class Label:
    def __init__(self, _id: int, supercategory: str, name: str):
        if _id is not None:
            self._id = _id
        if supercategory is not None:
            self.supercategory = supercategory
        if name is not None:
            self.name = name

    @classmethod
    def from_dict(cls, data):
        return cls(_id=data.get('id'),
                   supercategory=data.get('supercategory'),
                   name=data.get('name'))

    def to_dict(self):
        return self.__dict__


class Detection:
    def __init__(self, image_id: str, _id: str, file_name: str, coco_url: str, category_id: int, width: int,
                 height: int,
                 bbox: tuple, area: int):
        if image_id is not None:
            self.image_id = image_id
        if _id is not None:
            self.id = _id
        if file_name is not None:
            self.file_name = file_name
        if coco_url is not None:
            self.coco_url = coco_url
        if category_id is not None:
            self.category_id = category_id
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        if bbox is not None:
            self.bbox = bbox
        if area is not None:
            self.area = area

    @classmethod
    def from_dict(cls, data):
        return cls(image_id=data.get('image_id'),
                   _id=data.get('id'),
                   file_name=data.get('file_name'),
                   coco_url=data.get('coco_url'),
                   category_id=data.get('category_id'),
                   width=data.get('width'),
                   height=data.get('height'),
                   bbox=(round(data.get('bbox')[0]),
                         round(data.get('bbox')[1]),
                         round(data.get('bbox')[2]),
                         round(data.get('bbox')[3])),
                   area=round(data.get('area')))

    def to_dict(self):
        return self.__dict__


def pymongo_cursor_to_class_list(data, class_name):
    """
    Converts a pymongo cursor to a list of the given class
    :param data: pymongo cursor
    :param class_name: name of the class
    :return: list of objects of the given class
    """
    class_list = []
    for item in list(data):
        class_list.append(class_name.from_dict(item))
    return class_list


def pymongo_cursor_to_list(data):
    """
    Converts a pymongo cursor to a list
    :param data: pymongo cursor
    :return: list
    """
    data_list = []
    for item in list(data):
        data_list.append(item)
    return data_list


def pymongo_class_list_to_dict_list(data):
    """
    Converts a list of objects to a list of dictionaries
    :param data: list of objects
    :return: list of dictionaries
    """
    data_list = []
    for item in list(data):
        data_list.append(item.to_dict())
    return data_list


def find_objects_from_db(collection_name, class_name):
    """
    Returns a list of objects of the given class from the database
    :param collection_name: database collection name
    :param class_name: name of the class
    :return: list of objects of the given class
    """
    with MongoDBConnectionManager() as connection:
        return pymongo_cursor_to_class_list(connection.get_database('noise').get_collection(collection_name).find(),
                                            class_name)


def find_object_list_from_db(collection_name):
    """
    Returns a list from the database
    :param collection_name: database collection name
    :return: data list
    """
    with MongoDBConnectionManager() as connection:
        return pymongo_cursor_to_list(connection.get_database('noise').get_collection(collection_name).find())


def insert_dicts_to_db(collection_name, dicts: list[dict]):
    """
    Inserts a list of dictionaries to the database collection
    :param collection_name: database collection name
    :param dicts: list of dictionaries
    """
    with MongoDBConnectionManager() as connection:
        connection.get_database('noise').get_collection(collection_name).insert_many(dicts)


def insert_class_list_to_db(collection_name, class_list: list[object]):
    """
    Inserts a list of objects to the database collection
    :param collection_name: database collection name
    :param class_list: list of objects
    """
    with MongoDBConnectionManager() as connection:
        connection.get_database('noise').get_collection(collection_name).insert_many(
            pymongo_class_list_to_dict_list(class_list))
