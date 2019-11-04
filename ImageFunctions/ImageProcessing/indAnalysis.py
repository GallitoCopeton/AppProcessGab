import json
import math

import cv2
import numpy as np
import pandas as pd


from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ReadImages import readImage as rI


def readMask(url="../../Imagenes/mask_inv.png", size=90):
    mask = rI.readLocal(url)
    mask = resizeFixed(mask, size)
    mask = pP.contourBinarization(mask, 3, 3, 75, 4, inverse=False, mean=True)
    return mask


def resizeFixed(image, size=90):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def resizeAll(listOfImages, size=90):
    return [resizeFixed(image, size) for image in listResizetOfImages]


def andOperation(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def notOpetation(image):
    return cv2.bitwise_not(image)


def erosionDilation(imageBin, kSize):
    kernel = np.ones((kSize, kSize), np.uint8)
    erosion = cv2.erode(imageBin, kernel, iterations=1)
    return cv2.dilate(erosion, kernel, iterations=2)


# DEPRECATED


def quadrantAreaAnalysis(images):
    whitePixelsQuadrant = {}
    if len(images) > 4:
        images = imageQuadrantSplit(images)
    for i, image in enumerate(images):
        whitePixels = cv2.countNonZero(image)
        whitePixelsQuadrant['areaQ{}'.format(i+1)] = whitePixels
    return whitePixelsQuadrant

# DEPRECATED


def quadrantBlobAnalisys(images, bgrImages):
    measurementsDict = {
        'blobs': 0
    }
    if len(images) > 4:
        images = imageQuadrantSplit(images)
        bgrImages = imageQuadrantSplit(bgrImages)
    kernelOpen = np.ones((3, 3), np.uint8)

    for i, image in enumerate(images):
        # Data reset
        distance = 0
        quadrantArea = 0
        quadrantPerimeter = 0
        verticalDiag = 0
        horizontalDiag = 0
        # Open it once more to reduce noisy blobs
        imageOpen = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernelOpen)
        # Find contours and sort by the biggest
        contours, hierarchy = cv2.findContours(
            imageOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # If you find a single contour and it has more than 4 coordinates
        if len(contours) > 0 and len(contours[0]) > 4:
            contours = max(contours, key=cv2.contourArea)
            # Data extraction preparation => START

            # Max and min points
            extLeft = tuple(contours[contours[:, :, 0].argmin()][0])
            extRight = tuple(contours[contours[:, :, 0].argmax()][0])
            extTop = tuple(contours[contours[:, :, 1].argmin()][0])
            extBot = tuple(contours[contours[:, :, 1].argmax()][0])
            # Extract centroid of blob
            sideOfImage = image.shape[0]
            moments = cv2.moments(contours)
            centroidX = int(moments['m10']/moments['m00'])
            centroidY = int(moments['m01']/moments['m00'])
            if i == 0:
                fullImageCenterX = sideOfImage
                fullImageCenterY = sideOfImage
            elif i == 1:
                fullImageCenterX = 0
                fullImageCenterY = sideOfImage
            elif i == 2:
                fullImageCenterX = sideOfImage
                fullImageCenterY = 0
            elif i == 3:
                fullImageCenterX = 0
                fullImageCenterY = 0

            # Data extraction preparation => FINISH
            # Data extraction => START

            # Distance between blob and center of image in pixels
            distance = math.sqrt(
                (centroidX - fullImageCenterX)**2 + (centroidY - fullImageCenterY)**2)
            # Length of vertical diagonal in pixels
            verticalDiag = math.sqrt(
                (extBot[0] - extTop[0])**2 + (extBot[1] - extTop[1])**2)
            # Length of vertical diagonal in pixels
            horizontalDiag = math.sqrt(
                (extLeft[0] - extRight[0])**2 + (extLeft[1] - extRight[1])**2)
            quadrantArea = cv2.contourArea(contours)
            quadrantPerimeter = cv2.arcLength(
                contours, closed=True)

            # Data extraction => FINISH
            # Data storage
            measurementsDict['distanceQ{}'.format(i+1)] = distance
            measurementsDict['areaQ{}'.format(i+1)] = quadrantArea
            measurementsDict['perimeterQ{}'.format(i+1)] = quadrantPerimeter
            measurementsDict['verticalDiagQ{}'.format(i+1)] = verticalDiag
            measurementsDict['horizontalDiagQ{}'.format(i+1)] = horizontalDiag
            measurementsDict['blobs'] += 1
        # If the contour condition was not met, make every value of the dict 0
        else:
            measurementsDict['distanceQ{}'.format(i+1)] = distance
            measurementsDict['areaQ{}'.format(i+1)] = quadrantArea
            measurementsDict['perimeterQ{}'.format(i+1)] = quadrantPerimeter
            measurementsDict['verticalDiagQ{}'.format(i+1)] = verticalDiag
            measurementsDict['horizontalDiagQ{}'.format(i+1)] = horizontalDiag
            measurementsDict['blobs'] += 0
    return measurementsDict

# DEPRECATED


# DEPRECATED


def createNestedDataframes(markers, stats, data):
    dataFrames = []
    for i, marker in enumerate(markers):
        markerName = []
        for stat in stats:
            markerName.append(marker)
        arrays = [markerName, stats]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        if data:
            dataFrames.append(pd.DataFrame(data[:, i:i+2], columns=index))
        else:
            dataFrames.append(pd.DataFrame(columns=index))
        i += 3
    return dataFrames

# DEPRECATED


def createQuadrantDataframes(markers, quadrants, stats, data):
    dataFrames = []
    requiredLength = len(quadrants) * len(stats)
    for i, marker in enumerate(markers):
        quadrantsArray = np.array(
            quadrants*(int(requiredLength/len(quadrants))))
        quadrantsArray.sort()
        statsArray = np.array(stats*(int(requiredLength/len(stats))))

        arrays = [quadrantsArray, statsArray]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        if data is not None:
            dataFrames.append(pd.DataFrame(
                data[i:i+requiredLength, :], columns=index))
        else:
            dataFrames.append(pd.DataFrame(columns=index))
        i += requiredLength + 1
    return dataFrames


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
