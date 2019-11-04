import cv2


def adapBina(singleChannelImage, th1, th2, mean=True):
    thBin = cv2.THRESH_BINARY

    if(mean):
        return cv2.adaptiveThreshold(singleChannelImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thBin, th1, th2)
    else:
        return cv2.adaptiveThreshold(singleChannelImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     thBin, th1, th2)


def adapBinaInverse(singleChannelImage, th1, th2, mean=True):
    thBin = cv2.THRESH_BINARY_INV

    if(mean):
        return cv2.adaptiveThreshold(singleChannelImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thBin, th1, th2)
    else:
        return cv2.adaptiveThreshold(singleChannelImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     thBin, th1, th2)


def simpleBinarization(singleChannelImage, threshold):
    binarization = cv2.THRESH_BINARY
    _, binarizedImage = cv2.threshold(
        singleChannelImage, threshold, 255, binarization)
    return binarizedImage


def simpleBinarizationInverse(singleChannelImage, threshold):
    binarization = cv2.THRESH_BINARY_INV
    _, binarizedImage = cv2.threshold(
        singleChannelImage, threshold, 255, binarization)
    return binarizedImage


def otsuBinarize(image):
    _, binarizedImage = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binarizedImage


def otsuBinarizeInverse(image):
    _, binarizedImage = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return binarizedImage
