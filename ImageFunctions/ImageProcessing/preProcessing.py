import numpy as np
import cv2

from ImageFunctions.ImageProcessing import colorTransformations as cT


def gaussian(image, k, s):  # image and kernel size
    return cv2.GaussianBlur(image, (k, k), s)


def median(image, k):  # image and kernel size
    return cv2.medianBlur(image, k)


def normalizeLight(image):
    imageYUV = cT.BGR2YUV(image)
    imageYUV[:, :, 0] = cv2.equalizeHist(imageYUV[:, :, 0])
    return cv2.cvtColor(imageYUV, cv2.COLOR_YUV2BGR)


def adapHistogramEq(image, clipLimit=10.0, tileGridSize=(1, 1)):
    imageYUV = cT.BGR2YUV(image)
    clahe = cv2.createCLAHE(clipLimit=float(clipLimit),
                            tileGridSize=tileGridSize)
    imageYUV[:, :, 0] = clahe.apply(imageYUV[:, :, 0])
    return cv2.cvtColor(imageYUV, cv2.COLOR_YUV2BGR)


def clusterReconstruction(image, criteria, k, attempts):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    ret, label, centers = cv2.kmeans(
        Z, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    clusterReconstruction = centers[label.flatten()]
    return clusterReconstruction.reshape(
        (image.shape))
