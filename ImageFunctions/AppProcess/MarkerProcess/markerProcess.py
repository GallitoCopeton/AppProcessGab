import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ImageProcessing import indAnalysis as inA

scriptPath = os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath)

mask = inA.readMask(url='../../../Imagenes/mask_inv.png')


def clusteringProcess(listOfMarkers, kColors, attempts, extendedProcess=False):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100000, 100000)
    openKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    listOfNOTMarkers = []
    if extendedProcess:
        listOfLABMarkers = []
        listOfBinMarkers = []
        listOfTransMarkers = []
    for marker in listOfMarkers:
        # Preprocessing
        reconMarker = pP.clusterReconstruction(
            marker, criteria, kColors, attempts)
        LABMarker = cv2.cvtColor(reconMarker, cv2.COLOR_BGR2LAB)
        LABMarkerGray = pP.BGR2gray(LABMarker)
        binMarker = pP.otsuBinarize(LABMarkerGray)
        # Transformations and masking
        transMarker = inA.andOperation(binMarker, mask)
        transMarker = pP.applyTransformation(
            transMarker, openKernel, cv2.MORPH_OPEN)
        transMarker = pP.applyTransformation(
            transMarker, dilateKernel, cv2.MORPH_DILATE)
        NOTMarker = inA.notOpetation(transMarker)
        # Append marker to marker list
        listOfNOTMarkers.append(NOTMarker)
        if extendedProcess:
            listOfLABMarkers.append(LABMarkerGray)
            listOfBinMarkers.append(binMarker)
            listOfTransMarkers.append(transMarker)
    # Return lists of markers
    if extendedProcess:
        return [
            listOfLABMarkers,
            listOfBinMarkers,
            listOfTransMarkers,
            listOfNOTMarkers
        ]
    else:
        return listOfNOTMarkers


def oldProcess(listOfEqMarkers, extendedProcess=False):
    listOfNOTMarkers = []
    if extendedProcess:
        listOfBinMarkers = []
        listOfTransMarkers = []
    for markerEq in listOfEqMarkers:
        markerBin = pP.contourBinarizationOtsu(
            markerEq, 3, 3, 45, 3, Gs=0, inverse=False, mean=True)
        markerMasked = inA.andOperation(markerBin, mask)
        markerTrans = inA.erosionDilation(markerMasked, 3)
        markerNot = cv2.bitwise_not(markerTrans)
        listOfNOTMarkers.append(markerNot)
        if extendedProcess:
            listOfBinMarkers.append(markerBin)
            listOfTransMarkers.append(markerTrans)
    if extendedProcess:
        return [
            listOfEqMarkers,
            listOfBinMarkers,
            listOfTransMarkers,
            listOfNOTMarkers
        ]
    else:
        return listOfNOTMarkers
