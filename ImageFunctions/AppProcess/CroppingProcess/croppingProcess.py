import os
import sys

import cv2

from ImageFunctions.ImageProcessing import perspective as pPe
from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ImageProcessing import sorts as srt
from ImageFunctions.ImageProcessing import indAnalysis as inA

scriptPath = os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath)

mask = inA.readMask(url='../../../Imagenes/mask_inv.png')


def getEqTestSite(image):
    return pP.equalizeHistogram(getNonEqTestSite(image))


def getNonEqTestSite(image):
    testResized = pP.resizeImg(image, 728)
    testBin = pP.contourBinarization(
        testResized, 3, 7, 85, 2, inverse=True, mean=False)
    cardContours = pP.findTreeContours(testBin)
    for contour in cardContours:
        orderedContour = srt.sortPoints(contour)
        cardBin = pPe.perspectiveTransform(
            testBin, orderedContour, -5, binary=True)
        qrAndTestSiteContours = pP.findExternalContours(cardBin)
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
    height, width = testSite.shape[:2]
    markersContours = pP.findTreeContours(pP.contourBinarization(
        testSite, 3, 7, 85, 2, mean=False), 115000)
    if len(markersContours) == 5 or len(markersContours) == 7:
        markersContours = markersContours[1:]
    markersEq = []
    markers = []
    if(len(markersContours) == 4 or len(markersContours) == 6):
        srt.sortTests(markersContours)
        for i, markerContour in enumerate(markersContours):
            marker = pPe.getIndTest(testSite, markerContour)
            markers.append(marker)
    markers = inA.resizeAll(markers)
    return markers
