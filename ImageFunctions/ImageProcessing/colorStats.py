import json
import math

import cv2
import numpy as np
import pandas as pd

from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ReadImages import readImage as rI


def colorSegmentation(image, statsColor):  # Stats Hue and Saturation

    # Invert image BGR
    imageBGR_inv = ~image

    # Transform input inverse image to HSV
    imageHSV_inv = cv2.cvtColor(imageBGR_inv, cv2.COLOR_BGR2HSV)

    # Define boundaries for inRange
    cyan_cluster_mean = statsColor['H_mean']  # Mean
    cmean = cyan_cluster_mean
    cyan_cluster_std = statsColor['H_std']  # Standard Deviation
    # MPercentile 15
    non_cyan_exclude_saturation = statsColor['S_25_percentile']
    if(non_cyan_exclude_saturation < 0):
        non_cyan_exclude_saturation = 0  # Zero exception
    exc_sat_low = non_cyan_exclude_saturation
    non_cyan_exclude_value = 255
    exc_val_high = non_cyan_exclude_value
    percentage_of_cyan = 65
    cyan_cluster_dev = (3 * cyan_cluster_std / 100) * percentage_of_cyan
    cdev = cyan_cluster_dev

    # Make inRange operation
    mask3 = cv2.inRange(imageHSV_inv, np.array(
        [cmean-cdev, exc_sat_low, 10]), np.array([cmean+cdev, 255, exc_val_high]))

    # Closing operation
    mask3 = closing(mask3, 3)

    # Opening operation
    mask3 = opening(mask3, 3)

    # And operation between mask and image
    andOp = andOperation(image, mask3)
    ret, andOpbin = cv2.threshold(
        pP.BGR2gray(andOp), 1, 255, cv2.THRESH_BINARY)
    countPixels = cv2.countNonZero(andOpbin) / 50
    return (andOp, andOpbin, countPixels)  # color, binary, count


def controlStats(image, maskPath="../../Imagenes/mask3.png", maskSize=90, diab=False):
    stats = dict()
    mask3 = readMask(maskPath, size=maskSize)
    control = ~(andOperation(image, mask3))
    controlHSV = cv2.cvtColor(control, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(controlHSV)
    if(diab):
        h = h.flatten()[(h.flatten() >= 5) & (h.flatten() <= 250)]
    else:
        h = h.flatten()[(h.flatten() >= 50) & (h.flatten() <= 150)]

    s = s.flatten()[s.flatten() >= 15]
    v = v.flatten()[v.flatten() >= 10]
    # print("H: " + str(len(h)) + ", S: " + str(len(s)) + ", V: " + str(len(v)))
    channels = ['H', 'S', 'V']
    chVal = [h, s, v]
    for i, ch in enumerate(channels):
        stats[ch + '_mean'] = np.mean(chVal[i])
        stats[ch + '_median'] = np.median(chVal[i])
        stats[ch + '_std'] = np.std(chVal[i])
        stats[ch + '_25_percentile'] = np.percentile(chVal[i], 25)
    return stats


def grayControlStats(image, maskPath="../../Imagenes/mask3.png", maskSize=90):
    image = pP.BGR2gray(image)
    stats = dict()
    mask3 = readMask(maskPath, size=maskSize)
    control = andOperation(image, mask3)
    # c = control.flatten()[(control.flatten() >= 5) & (control.flatten() <= 255)]
    c = control.flatten()[(control.flatten() >= 15)
                          & (control.flatten() <= 235)]
    stats['G_mean'] = np.mean(c)
    stats['G_median'] = np.median(c)
    stats['G_std'] = np.std(c)
    stats['G_25_percentile'] = np.percentile(c, 25)
    return stats


def colorStatsXMarker(image):
    statsMeasures = ['Mean', 'Median', 'Std', 'Percentile25']
    statsXMarker = dict()
    statsH, statsS = controlStats(image)
    statsH, statsS = controlStats(image)
    for i, h in enumerate(statsH):
        statsXMarker['H_' + statsMeasures[i]] = h
        statsXMarker['S_' + statsMeasures[i]] = statsS[i]
    return statsXMarker


def isEmpty(imageBinary):
    rows, cols = imageBinary.shape
    totalPx = rows * cols
    whitePx = cv2.countNonZero(imageBinary)
    percentage = ((totalPx - whitePx) * 100) / totalPx
    return (percentage < 35)


def closing(imageBin, kSize):
    kernel = np.ones((kSize, kSize), np.uint8)
    erosion = cv2.erode(imageBin, kernel, iterations=1)
    return cv2.dilate(erosion, kernel, iterations=1)


def opening(imageBin, kSize):
    kernel = np.ones((kSize, kSize), np.uint8)
    dilate = cv2.dilate(imageBin, kernel, iterations=1)
    return cv2.erode(dilate, kernel, iterations=1)


def totalWhitePixels(image):
    return cv2.countNonZero(image)
