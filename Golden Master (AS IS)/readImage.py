import base64

import cv2
import numpy as np
import pymongo as pymongo

# Read only, mongodb connection
MONGO_URL = 'mongodb://findOnlyReadUser:RojutuNHqy@unimahealth.ddns.net:888/datamap'
MONGO_CLOUD_URL = 'mongodb+srv://findOnlyReadUser:RojutuNHqy@clusterfinddemo-lwvvo.mongodb.net/datamap?retryWrites=true'


def readb64(base64_string):
    nparr = np.fromstring(base64.b64decode(base64_string), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Read from imagestotals


def readFromDB(qr, array, localDB, count=0):
    if localDB is True:
        client = pymongo.MongoClient(MONGO_URL)
    else:
        client = pymongo.MongoClient(MONGO_CLOUD_URL)
    db = client.datamap

    # One image or a list of images
    if (array):
        cursor_img_totals = db.imagestotals.find({'fileName': qr})
    else:
        cursor_img_totals = db.imagestotals.find_one(
            {'fileName': qr, 'count': count})

    # Colection images totals
    if(cursor_img_totals):
        if(array):
            return imageArray(cursor_img_totals)
        else:
            return readb64(cursor_img_totals['file'])

    # Colections images
    else:
        cursor_img = db.images.find_one({'fileName': qr})
        if(cursor_img):
            cvimg = readb64(cursor_img['file'])
            return cvimg
        else:
            return ("No records!")

# Return array of base64 images


def imageArray(cursor):
    images = []
    for c in cursor:
        images.append(readb64(c['file']))
    return images


def readLocal(path):
    return cv2.imread(path, 1)


def readImage(qr, path='../../Imagenes/', count=0, local=False, array=False, localDB=False, ext='png'):
    if(local):
        img = readLocal(path + str(qr) + "." + ext)
        return img
    else:
        return readFromDB(qr, array, count, localDB)
