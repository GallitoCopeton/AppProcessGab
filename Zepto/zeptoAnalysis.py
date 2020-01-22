# %%
import datetime
import json
import os
import re

import cv2
import numpy as np
import pandas as pd

import qrQuery
import ShowProcess.showProcesses as sP
from AppProcess.CroppingProcess import croppingProcess as cP
from AppProcess.MarkerProcess import markerProcess
from ImageProcessing import binarizations as bZ
from ImageProcessing import colorTransformations as cT
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA
from ReadImages import readImage as rI
# %%
with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
# %%
# Real: base con diagnósticos reales Zepto: base de pruebas de Zeptometrix Clean: base con marcadores seleccionados
zeptoConnection = connections['zepto']
zeptoImagesCollection = qrQuery.getCollection(
    zeptoConnection['URI'], zeptoConnection['databaseName'], zeptoConnection['collections']['imagesCollectionName'])
zeptoDataCollection = qrQuery.getCollection(
    zeptoConnection['URI'], zeptoConnection['databaseName'], zeptoConnection['collections']['dataCollectionName'])
# %%
# Image processing tools
mask = bZ.otsuBinarize(markerProcess.fullMask)
mask = iO.applyTransformation(mask, np.ones((3, 3)), cv2.MORPH_ERODE, 2)
k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kColors = 3
attempts = 6

# Info of the markers we want to analyze
features2Extract = ['nBlobs', 'totalArea', 'fullBlobs', 'bigBlobs', 'medBlobs', 'smallBlobs', 'q0HasBlob',
                    'q1HasBlob', 'q2HasBlob', 'q3HasBlob', 'noise', 'distance', 'distanceBetweenPoints', 'diagnostic']
# Markers for table
markerNames = ['ESAT6', 'RV1681', 'CFP10', 'Control']
# Table's columns
iterables = [markerNames, features2Extract]
index = pd.MultiIndex.from_product(iterables, names=['Protein', 'Info'])
# Query
limit = 3
zeptoTests = zeptoImagesCollection.find().limit(limit)

# Image decoder


def fixImage(image): return iO.resizeImg(rI.readb64(image['file']), 728)


markersDict = {}
for test in zeptoTests:
    qr = test['fileName']
    count = test['count']
    image = fixImage(test)
    try:
        testSite = cP.getTestArea(image)
        markerImages = cP.getMarkers(testSite)
    except:
        continue
    register = zeptoDataCollection.find_one({'qrCode': qr, 'count': count})
    for i, markerInfo in enumerate(zip(markerImages, register['marker'])):
        markerImage, markerRegister = markerInfo
        name = markerRegister['name']
        try:
            diagnostic = markerRegister['result'].upper()
            diagnostic = 1 if diagnostic == 'Positive' else 0
        except Exception:
            continue
        # Fourier processing
        fourierMarker = markerProcess.fourierProcess([markerImage])[0]
        markerGray = cT.BGR2gray(markerImage)
        noise = cv2.subtract(np.float32(markerGray),
                             fourierMarker, dtype=cv2.CV_32F).sum()
        # Blood only extraction for agglutination processing
        processedMarkers = markerProcess.clusteringProcess(
            [markerImage], 3, 6, True)
        clusteredMarker = processedMarkers[-1][0]
        reconMarker = processedMarkers[1][0]
        uniqueValues = np.unique(
            reconMarker.reshape(-1, 1), axis=0)
        reconMarker[reconMarker[:, :] != uniqueValues[1]] = 0
        reconMarker = iO.applyTransformation(reconMarker, k, cv2.MORPH_OPEN, 1)
        reconMarker = iO.applyTransformation(
            reconMarker, k, cv2.MORPH_DILATE, 1)
        reconMarkerMask = bZ.otsuBinarize(
            iO.andOperation(reconMarker, mask), 1)
        reconMarkerMask = iO.applyTransformation(
            reconMarkerMask, k, cv2.MORPH_DILATE, 1)
        bloodMasked = markerImage.copy()
        bloodMasked = bloodMasked[:, :, 0]
        bloodMasked = iO.andOperation(bloodMasked, reconMarkerMask)
        laplacian = cv2.Laplacian(bloodMasked, cv2.CV_64F)
        laplacian[laplacian[:, :] > 60] = 0
        lapBina = bZ.simpleBinarization(laplacian, 1)
        aglutinQuad = inA.imageQuadrantSplit(lapBina)
        qRects = []
        areas = 0
        distances = 0
        distancesBetweenPoints = 0
        for nQ, q in enumerate(aglutinQuad):
            q = q.astype(np.uint8)
            qCoords = []
            rows = q.shape[0]
            cols = q.shape[1]
            for row in range(0, rows):
                for col in range(0, cols):
                    if q[col, row] > 0:
                        qCoords.append((col, row))
            xCoords = [x[0] for x in qCoords]
            yCoords = [y[1] for y in qCoords]
            dx = np.diff(xCoords).std()
            dy = np.diff(yCoords).std()
            distancesBetweenPoints += sum(
                inA.getDistanceBetweenPoints(qCoords))
            _, cnts, _ = cv2.findContours(
                q, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
                if(cv2.contourArea(c) > 2):
                    areas += cv2.contourArea(c)
                    distances += inA.getContourDistance(c, nQ)
        if qr not in markersDict.keys():
            markersDict[qr] = {}
        markerNot = markerProcess.clusteringProcess(
            [markerImage], kColors, attempts, extendedProcess=False)[0]
        sP.showImage(fourierMarker, figSize=(3, 3), title=name)
        features = list(inA.extractFeatures(
            markerNot, features2Extract).values())
        features.append(noise)
        features.append(distances)
        features.append(distancesBetweenPoints)
        features.append(diagnostic)
        if count not in markersDict[qr].keys():
            markersDict[qr][count] = {}
        markersDict[qr][count][name] = features

realFullDf = []
for qr in markersDict.keys():
    for count in markersDict[qr].keys():
        markerInfoFull = []
        for proteinName in markersDict[qr][count].keys():
            markerInfoFull.append(markersDict[qr][count][proteinName])
        markerInfoFull = np.array(markerInfoFull).ravel()
        df = pd.DataFrame(markerInfoFull.reshape(
            1, len(markerInfoFull)), columns=index)
        realFullDf.append(df)
realFullDf = pd.concat(realFullDf)
# %% EXCEL CREATION
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
dfsFolder = '../Feature Tables'
todaysDate = datetime.datetime.now()
dateString = re.sub(r':', '_', todaysDate.ctime())[4:-5]
currentFolder = os.path.dirname(os.path.realpath(__file__))
folderPath = f'{currentFolder}/{dfsFolder}'
qrQuery.makeFolders(folderPath)
infoCsvName = '/'.join([folderPath, f'Dataframe de {dateString}.xlsx'])
realFullDf.to_excel(infoCsvName, index=True)
