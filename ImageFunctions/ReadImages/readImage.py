import base64
import os
from datetime import datetime

import cv2
import numpy as np
import pymongo
from matplotlib import pyplot as plt


def readLocal(path):
    return cv2.imread(path, 1)


def readSingleFromDb(collection, qr, count=0):
    """
    This function will read images from a database, local or remote.
    Will return 0 if no images found
    """
    cursor = collection.find_one({'filename': qr, 'count': count})
    if cursor is not None:
        return readb64(cursor['file'])
    else:
        return 0


def readSingleFromDbDetails(collection, qr, count=0):
    """
    This function will read images from a database, local or remote.
    Will return 0 if no images found
    """
    cursor = collection.find_one({'filename': qr, 'count': count})
    if cursor is not None:
        return {
            'image': readb64(cursor['file']),
            'createdAt': datetime.date(cursor['createdAt']),
            'qr': cursor['fileName'],
            'count': cursor['count']
        }
    else:
        return 0


def readManyFromDb(collection, qr):
    """
    This function will read images from a database, local or remote.
    Will return an empty array if no images found
    """
    cursor = collection.find({'filename': qr}).sort('_id', -1)
    if cursor is not None:
        return getImageList(cursor)
    else:
        return []


def readManyFromDbDetails(collection, qr):
    """
    This function will read images from a database, local or remote.
    Will return an empty array if no images found
    """
    cursor = collection.find({'filename': qr}).sort('_id', -1).limit(10)
    if cursor is not None:
        return getImageListDetails(cursor)
    else:
        return []


def customQuery(collection, query, limit=10):
    """
    This function will read images from a database, local or remote.
    Will return an empty array if no images found
    """
    cursor = collection.find(query).sort('_id', -1).limit(limit)
    if cursor is not None:
        return getImageList(cursor)
    else:
        return []


def customQueryDetails(collection, query, limit=10):
    """
    This function will read images from a database, local or remote.
    Will return an empty array if no images found
    """
    cursor = collection.find(query).sort('_id', -1).limit(limit)
    if cursor is not None:
        return getImageListDetails(cursor)
    else:
        return []


def readb64(base64_string):
    """
    This function will decode images from base 64 to matrix form.
    """
    nparr = np.fromstring(base64.b64decode(base64_string, '-_'), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rows, cols, _ = image.shape
    if(rows < cols):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def getImageList(cursor):
    """
    This function will arange a list of decoded images from a Mongo cursor.
    """
    return [readb64(entry['file']) for entry in cursor]


def getImageListDetails(cursor):
    """
    This function will arange a list of decoded images from a Mongo cursor.
    """
    imagesInfo = []
    for entry in cursor:
        info = {}
        if 'file' in entry.keys():
            info['image'] = readb64(entry['file'])
        if 'createdAt' in entry.keys():
            info['createdAt'] = datetime.date(entry['createdAt'])
        if 'fileName' in entry.keys():
            info['qr'] = entry['fileName']
        if 'count' in entry.keys():
            info['count'] = entry['count']
        imagesInfo.append(info)
    return imagesInfo
