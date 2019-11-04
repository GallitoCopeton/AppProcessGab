import os

import cv2

from ImageFunctions.ImageProcessing import perspective as pPe
from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ImageProcessing import sorts as srt
from ImageFunctions.ImageProcessing import indAnalysis as inA
from ImageFunctions.ImageProcessing import colorTransformations as cT
from ImageFunctions.ImageProcessing import binarizations as bZ
from ImageFunctions.ImageProcessing import imageOperations as iO
from ImageFunctions.ImageProcessing import contours as ctr

scriptPath = os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath)

mask = inA.readMask(url='../../../Imagenes/mask_inv.png')


def getEqTestSite(image):
    return pP.adapHistogramEq(getNonEqTestSite(image))


def getNonEqTestSite(image):
    testResized = iO.resizeImg(image, 728)
    testGray = cT.BGR2gray(testResized)
    testBlur = pP.gaussian(testGray, k=3, s=0)
    testBlur = pP.median(testBlur, 7)
    testBin = bZ.adapBinaInverse(testBlur, 85, 2, mean=False)
    cardContours = ctr.findTreeContours(testBin)
    for contour in cardContours:
        orderedContour = srt.sortPoints(contour)
        cardBin = pPe.perspectiveTransform(
            testBin, orderedContour, -5, binary=True)
        qrAndTestSiteContours = ctr.findExternalContours(cardBin)
        if len(qrAndTestSiteContours) == 2:
            card = pPe.perspectiveTransform(
                testResized, orderedContour, -5)
            contour1, contour2 = qrAndTestSiteContours
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            if area1 > area2:
                testSiteContours = contour1
                qrSiteContours = contour2
            else:
                testSiteContours = contour2
                qrSiteContours = contour1
            testSiteContoursOrdered = srt.sortPoints(testSiteContours)
            qrSiteContoursOrdered = srt.sortPoints(qrSiteContours)
            if qrSiteContoursOrdered[0][0][0] > testSiteContoursOrdered[0][0][0] and qrSiteContoursOrdered[2][0][1] > testSiteContoursOrdered[2][0][1]:
                testSite = pPe.perspectiveTransform(
                    card, testSiteContoursOrdered, offset=5)
    return testSite


def getMarkers(testSite):
    testSiteGray = cT.BGR2gray(testSite)
    testSiteBlur = pP.gaussian(testSiteGray, k=3, s=0)
    testSiteBlur = pP.median(testSiteBlur, 7)
    testSiteBin = bZ.adapBinaInverse(testSiteBlur, 85, 2, mean=False)
    markersContours = ctr.findTreeContours(testSiteBin, 115000)
    if len(markersContours) == 5 or len(markersContours) == 7:
        markersContours = markersContours[1:]
    markers = []
    if(len(markersContours) == 4 or len(markersContours) == 6):
        srt.sortTests(markersContours)
        markers = iO.resizeAll([pPe.getIndTest(testSite, markerContour)
                                for markerContour in markersContours])
    return markers
