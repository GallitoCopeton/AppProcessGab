import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ImageProcessing import indAnalysis as inA
from ImageFunctions.ImageProcessing import colorTransformations as cT
from ImageFunctions.ImageProcessing import binarizations as bZ
from ImageFunctions.ImageProcessing import imageOperations as iO
from ImageFunctions.ImageProcessing import contours as ctr

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
        LABMarkerGray = cT.BGR2gray(LABMarker)
        binMarker = bZ.otsuBinarize(LABMarkerGray)
        # Transformations and masking
        transMarker = iO.andOperation(binMarker, mask)
        transMarker = iO.applyTransformation(
            transMarker, openKernel, cv2.MORPH_OPEN)
        transMarker = iO.applyTransformation(
            transMarker, dilateKernel, cv2.MORPH_DILATE)
        NOTMarker = iO.notOpetation(transMarker)
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
        markerGray = cT.BGR2gray(markerEq)
        markerBlur = pP.gaussian(markerGray, k=3, s=0)
        markerBin = bZ.adapBina(markerBlur, 85, 2, mean=True)
        markerMasked = iO.andOperation(markerBin, mask)
        markerTrans = iO.erosionDilation(markerMasked, 3)
        markerNot = iO.notOpetation(markerTrans)
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
