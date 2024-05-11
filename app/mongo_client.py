from pymongo import MongoClient


# from Detection import Detection
# from Label import Label


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


# def pymongo_cursor_to_objects(data):
#     users = []
#     for item in list(data):
#         users.append(Detection.from_dict(item))
#     return users


def pymongo_cursor_to_labels(data, class_name):
    users = []
    for item in list(data):
        users.append(class_name.from_dict(item))
    return users


# todo delete
def get_database(database_name, collection_name):
    client = MongoClient("mongodb://localhost:27017/")
    return client.get_database(database_name).get_collection(collection_name)


# todo передать класс как параметр
# def find_objects_from_db(collection_name):
#     with MongoDBConnectionManager() as connection:
#         return pymongo_cursor_to_objects(connection.get_database("noise").get_collection(collection_name).find())


def find_objects_from_db_label(collection_name, class_name):
    with MongoDBConnectionManager() as connection:
        return pymongo_cursor_to_labels(connection.get_database("noise").get_collection(collection_name).find(),
                                        class_name)


# def find_objects_from_db_with_category_ids(collection_name, category_ids):
# with MongoDBConnectionManager() as connection:
#     db = connection.get_database("noise").get_collection(collection_name)
#     return pymongo_cursor_to_objects(db.find({'category_id': {'$in': category_ids}}))


def insert_dicts_to_db(db, dicts):
    db.insert_many(dicts)


def insert_objects_to_db(db, objects):
    objects_dict = []
    for item in objects:
        objects_dict.append(item.to_dict())
    db.insert_many(objects_dict)
