import json
import math

import cv2
import numpy as np
import pandas as pd


from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ImageProcessing import binarizations as bZ
from ImageFunctions.ImageProcessing import imageOperations as iO
from ImageFunctions.ImageProcessing import colorTransformations as cT
from ImageFunctions.ReadImages import readImage as rI


def readMask(url="../../Imagenes/mask_inv.png", size=90):
    mask = rI.readLocal(url)
    mask = iO.resizeFixed(mask, size)
    maskGray = cT.BGR2gray(mask)
    mask = bZ.otsuBinarize(maskGray)
    return mask


def imageQuadrantSplit(image):
    halfWidth = int(image.shape[0]/2)
    halfHeight = int(image.shape[1]/2)
    if len(image.shape) == 3:
        firstQuadrant = image[0:halfWidth, 0:halfHeight, :]
        secondQuadrant = image[0:halfWidth, halfHeight:, :]
        thirdQuadrant = image[halfWidth:, 0:halfHeight, :]
        fourthQuadrant = image[halfWidth:, halfHeight:, :]
    else:
        firstQuadrant = image[0:halfWidth, 0:halfHeight]
        thirdQuadrant = image[halfWidth:, 0:halfHeight]
        secondQuadrant = image[0:halfWidth, halfHeight:]
        fourthQuadrant = image[halfWidth:, halfHeight:]
    return [firstQuadrant, secondQuadrant, thirdQuadrant, fourthQuadrant]


def mergeQuadrants(quadrants):
    q1, q2, q3, q4 = quadrants
    q1q2 = cv2.hconcat([q1, q2])
    q3q4 = cv2.hconcat([q3, q4])
    return cv2.vconcat([q1q2, q3q4])


def hasConvexDefect(binaryMarker):
    hasDefect = False
    kernel = np.ones((5, 5), np.uint8)
    openMarker = cv2.morphologyEx(binaryMarker, cv2.MORPH_OPEN, kernel)
    dilatedMarker = cv2.morphologyEx(
        openMarker, cv2.MORPH_DILATE, kernel)
    distTransMarker = cv2.distanceTransform(
        dilatedMarker, cv2.DIST_L2, 5)
    _, sureForegroundMarker = cv2.threshold(
        distTransMarker, .19*distTransMarker.max(), 255, cv2.THRESH_BINARY)
    sureForegroundMarker = sureForegroundMarker.astype(np.uint8)
    xContours, _ = cv2.findContours(
        sureForegroundMarker, 2, cv2.CHAIN_APPROX_SIMPLE)
    xContour = max(xContours, key=cv2.contourArea)
    hull = cv2.convexHull(xContour, returnPoints=False)
    defects = cv2.convexityDefects(xContour, hull)
    def sortFunc(defect): return defect[0][3]
    defects = sorted(defects, key=sortFunc, reverse=True)[0:4]
    points = []
    for i in range(len(defects)):
        s, e, f, d = defects[i][0]
        far = tuple(xContour[f][0])
        points.append(far)
    sortedPoints = sorted(points, key=lambda x: x[0])
    up, right, left, down = sortedPoints
    vertDist = math.sqrt((up[0]-down[0])**2 + (up[1]-down[1])**2)
    horDist = math.sqrt((right[0]-left[0])**2 + (right[1]-left[1])**2)
    distances = [vertDist, horDist]
    distanceDifference = max(distances)-min(distances)
    if distanceDifference > 2.5:
        hasDefect = True
        return hasDefect, distanceDifference
    else:
        return hasDefect, distanceDifference


def getBackgroundColor(markerGray, kSize, distance):
    h = markerGray.shape[1]
    percentage = int(distance*h)
    corner1 = markerGray[percentage:percentage +
                         kSize, percentage:percentage+kSize]
    corner1Max = corner1.max()
    corner1Min = corner1.min()
    corner1Mean = np.array([corner1Max, corner1Min]).mean()
    corner2 = markerGray[percentage:percentage +
                         kSize, h-percentage-kSize:h-percentage]
    corner2Max = corner2.max()
    corner2Min = corner2.min()
    corner2Mean = np.array([corner2Max, corner2Min]).mean()
    corner3 = markerGray[h-percentage-kSize:h -
                         percentage, percentage:percentage+kSize]
    corner3Max = corner3.max()
    corner3Min = corner3.min()
    corner3Mean = np.array([corner3Max, corner3Min]).mean()
    corner4 = markerGray[h-percentage-kSize:h -
                         percentage, h-percentage-kSize:h-percentage]
    corner4Max = corner4.max()
    corner4Min = corner4.min()
    corner4Mean = np.array([corner4Max, corner4Min]).mean()
    return np.array([corner1Mean, corner2Mean, corner3Mean, corner4Mean])
