import cv2
import numpy as np


def resizeImg(image, width):
    height_res = image.shape[0]
    width_res = image.shape[1]
    scale = (height_res*width)/width_res
    return cv2.resize(image, (width, int(scale)), interpolation=cv2.INTER_AREA)


def erosionDilation(binaryImage, kSize):
    kernel = np.ones((kSize, kSize), np.uint8)
    erosion = cv2.erode(binaryImage, kernel, iterations=1)
    return cv2.dilate(erosion, kernel, iterations=2)


def resizeFixed(image, size=90):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def resizeAll(listOfImages, size=90):
    return [resizeFixed(image, size) for image in listOfImages]


def applyTransformation(binaryImage, kernel, operation):
    return cv2.morphologyEx(binaryImage, operation, kernel)


def closing(binaryImage, kSize):
    kernel = np.ones((kSize, kSize), np.uint8)
    erosion = cv2.erode(binaryImage, kernel, iterations=1)
    return cv2.dilate(erosion, kernel, iterations=1)


def opening(binaryImage, kSize):
    kernel = np.ones((kSize, kSize), np.uint8)
    dilate = cv2.dilate(binaryImage, kernel, iterations=1)
    return cv2.erode(dilate, kernel, iterations=1)


def andOperation(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def notOpetation(image):
    return cv2.bitwise_not(image)
