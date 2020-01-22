# %%
import datetime
import os
import re

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import qrQuery
from AppProcess.MarkerProcess import markerProcess
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA
from ImageProcessing import colorTransformations as cT
from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
from ImageProcessing import binarizations as bZ


def fixMarker(marker): return iO.resizeFixed(rI.readb64(marker['image']))


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# %% Collections
realURI = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
realDbName = 'prodLaboratorio'
realCollectionName = 'markerTotals'
realMarkerCollection = qrQuery.getCollection(
    realURI, realDbName, realCollectionName)

cleanURI = 'mongodb://findUser:85d4s32D2%23diA@idenmon.zapto.org:888/?authSource=testerSrv'
cleanDbName = 'testerSrv'
cleanCollectionName = 'cleanMarkerTotals'
cleanCollection = qrQuery.getCollection(
    cleanURI, cleanDbName, cleanCollectionName)
# %%
# Info of the markers we want to analyze
mask = bZ.otsuBinarize(cT.BGR2gray(
    rI.readLocal('../Smoothness quantifying/mask.png')))
mask = iO.applyTransformation(mask, np.ones((3, 3)), cv2.MORPH_ERODE, 1)
markerNamesReal = ['P24']
features2Extract = ['nBlobs', 'totalArea', 'fullBlobs', 'bigBlobs', 'medBlobs',
                    'smallBlobs', 'q0HasBlob', 'q1HasBlob', 'q2HasBlob', 'q3HasBlob', 'diagnostic']
# Query: markers I want, that their diagnostic exists
markersP = realMarkerCollection.find(
    {'marker': {'$in': markerNamesReal}, 'diagnostic': 'P'}, no_cursor_timeout=True).limit(60).sort('_id', 1)
markersN = realMarkerCollection.find(
    {'marker': {'$in': markerNamesReal}, 'diagnostic': 'N'}, no_cursor_timeout=True).limit(60).sort('_id', 1)
# %%
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kColors = 3
attempts = 2
for markerP, markerN in zip(markersP, markersN):
    print('*'*20)
    diagP = markerP['diagnostic']
    diagN = markerN['diagnostic']
    markerP = fixMarker(markerP)
    markerN = fixMarker(markerN)
    markerP = cT.BGR2RGB(markerP)
    markerN = cT.BGR2RGB(markerN)
    markerP = markerProcess.clusteringProcess(
        [markerP], kColors, attempts, True)[1][0]
    markerN = markerProcess.clusteringProcess(
        [markerN], kColors, attempts, True)[1][0]
    markerP = cv2.medianBlur(markerP, 3)
    markerN = cv2.medianBlur(markerN, 3)
    markerP = iO.andOperation(markerP, mask)
    markerN = iO.andOperation(markerN, mask)
    print(np.unique(
        markerN.reshape(-1, 1), axis=0))
    colors = ("r", "g", "b")
#    colors = ("l", "a", "b")
    featuresN = []
    featuresP = []
    chansP = cv2.split(markerP)
    chansN = cv2.split(markerN)
    # loop over the image channels
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axP = axs[0][0]
    axP.set_title('P')
    axP.set_xlim(80, 170)
    axP.set_ylim(-10, 1900)
    axN = axs[1][0]
    axN.set_title('N')
    axN.set_xlim(80, 170)
    axN.set_ylim(-10, 1900)
    axMP = axs[0][1]
    axMP.set_title('P marker')
    axMN = axs[1][1]
    axMN.set_title('N marker')
    for (chanP, chanN, color) in zip(chansP, chansN, colors):
        histP = cv2.calcHist([chanP], [0], None, [256], [0, 256])
        histN = cv2.calcHist([chanN], [0], None, [256], [0, 256])
        featuresP.extend(histP)
        featuresN.extend(histN)
        axP.plot(histP, color)
        axN.plot(histN, color)
        print(sorted(list(set(histP.ravel()))))
        print(sorted(list(set(histN.ravel()))))
        print(sorted(list(set(histP.ravel())))
              [-2] - sorted(list(set(histP.ravel())))[-4])
        print(sorted(list(set(histN.ravel())))
              [-2] - sorted(list(set(histN.ravel())))[-4])
        print(sorted(list(set(histP.ravel())))
              [-4] / sorted(list(set(histP.ravel())))[-2])
        print(sorted(list(set(histN.ravel())))
              [-4] / sorted(list(set(histN.ravel())))[-2])
        # print(list(set(histN.ravel()))[-1] - list(set(histN.ravel()))[2])
    axMP.imshow(markerP)
    axMN.imshow(markerN)
    plt.show()
    print('*'*20)
