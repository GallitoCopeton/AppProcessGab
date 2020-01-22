# %%

import datetime
import os
import re
import sys
import math
import operator
import tensorflow as tf
import pymongo as pymongo
import base64
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import qrQuery
from AppProcess.MarkerProcess import markerProcess
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA2
from ImageProcessing import colorTransformations as cT
from ImageProcessing import blobAnalysis as bA
from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
from ImageProcessing import binarizations as bZ
from machineLearningUtilities import modelPerformance as mP

#sys.path.insert(0, '../Golden Master (AS IS)')
#import indAnalysis as inA
#import perspective as pPe
#import sorts as srt
#import preProcessing as pP
#import readImage
def fixMarker(marker): return iO.resizeFixed(rI.readMarkerb64(marker['image']))


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# Scripts para leer y procesar imagen



def load_graph(frozen_graph_filename):
    # Load file from disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


graph = load_graph('../data/opt_unimaHealth.pb')
# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/x:0')
y = graph.get_tensor_by_name('prefix/y_pred:0')
pkeep = graph.get_tensor_by_name('prefix/pkeep:0')
# %% Collections
realURI = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
realDbName = 'prodLaboratorio'
realCollectionName = 'markerTotals'
realMarkerCollection = qrQuery.getCollection(
    realURI, realDbName, realCollectionName)

cleanURI = 'mongodb://findUser:85d4s32D2%23diA@idenmon.zapto.org:888/?authSource=oldProcesserSrv'
cleanDbName = 'oldProcesserSrv'
cleanCollectionName = 'cleanMarkerTotals'
cleanCollection = qrQuery.getCollection(
    cleanURI, cleanDbName, cleanCollectionName)

zeptoURI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
zeptoDbName = 'findValidation'
newMarkerCollectionName = 'markerTotals'
newMarkerCollection = qrQuery.getCollection(zeptoURI, zeptoDbName, newMarkerCollectionName)
# %%
# Info of the markers we want to analyze
markerNamesReal = ['RV1681', 'ESAT6', 'P24', 'CFP10']
features2Extract = ['nBlobs', 'totalArea', 'fullBlobs', 'bigBlobs', 'medBlobs',
                    'smallBlobs', 'q0HasBlob', 'q1HasBlob', 'q2HasBlob', 'q3HasBlob', 'diagnostic']
# Query: markers I want, that their diagnostic exists
limit = 0
markers = newMarkerCollection.find(
    {'diagnostic': {'$ne': None}}, no_cursor_timeout=True).limit(limit).sort('_id', -1)
blobMarkers = []
listoldProcesssBinary = []
pkeepConst = 1
# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/x:0')
y = graph.get_tensor_by_name('prefix/y_pred:0')
pkeep = graph.get_tensor_by_name('prefix/pkeep:0')
AREAS_DIAM_TH = 20
CNN_PROB_TH = 0.7
tPcNN = 0
tNcNN = 0
fPcNN = 0
fNcNN = 0

tPaNN = 0
tNaNN = 0
fPaNN = 0
fNaNN = 0
k = np.ones((2,2))
mask = bZ.otsuBinarize(cT.BGR2gray(
    rI.readLocal('../Smoothness quantifying/mask.png')))
mask = iO.applyTransformation(mask, np.ones((3, 3)), cv2.MORPH_ERODE, 2)
# %%
modelsFolder = './'
fileNames = ['ANN_92.pkl']
means = [ 2.69491525e+00,  1.58893490e+01,  7.30932203e-02,  2.26694915e-01,
        8.19915254e-01,  1.57521186e+00,  6.51483051e-01,  5.86864407e-01,
        6.67372881e-01,  6.71610169e-01, -7.91089033e+08,  5.84200212e+02]
variances = [2.35395720e+00, 2.60888031e+02, 1.05886195e-01, 3.00304331e-01,
       1.13706101e+00, 1.88205504e+00, 2.27052885e-01, 2.42454575e-01,
       2.21986319e-01, 2.20549950e-01, 1.65927325e+15, 2.19430584e+04]
models = [mP.loadModel(modelsFolder, fileName) for fileName in fileNames]
for i, marker in enumerate(markers):
    print('*'*60)
    name = marker['marker']
    count = marker['count']
    diag = marker['diagnostic']
    print(f'Real diag: {diag}')
    try:
        marker = fixMarker(marker)
    except: 
        nparr = np.frombuffer(base64.b64decode(marker['image']), np.uint8)
        marker = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # OLD IMAGE PROCESSING
#    oldProcessBin = pP.contourBinarizationOtsu(
#        marker, 3, 3, 45, 4, Gs=0, inverse=False, mean=True)
#    oldProcessBinMask = inA.andOperation(oldProcessBin, mask)
#    oldProcess = inA.erosionDilation(oldProcessBinMask, 3)
#    oldProcess = cv2.bitwise_not(oldProcess)
#    oldBlobs = inA.blobDetect(oldProcess)
#    oldArea = inA.areaEstimation(oldBlobs)
    # CNN PROCESSING
    marker32 = marker.astype('float32')
    marker32 = np.multiply(marker32, 1.0/255.0)
    x_batch = marker32.reshape(1, 90, 90, 3)
    # FOURIER PROCESSING
    fourierMarker = markerProcess.fourierProcess([marker])[0]
    markerGray = cT.BGR2gray(marker)
    noise = cv2.subtract(np.float32(markerGray), fourierMarker, dtype=cv2.CV_32F).sum()
    # CLUSTER PROCESSING
    processedMarkers = markerProcess.clusteringProcess([marker], 3, 6, True)
    clusteredMarker = processedMarkers[-1][0]
    reconMarker = processedMarkers[1][0]
    uniqueValues = np.unique(
            reconMarker.reshape(-1, 1), axis=0)
    reconMarker[reconMarker[:,:] != uniqueValues[1]] = 0
    reconMarker = iO.applyTransformation(reconMarker, k, cv2.MORPH_OPEN, 1)
    reconMarker = iO.applyTransformation(reconMarker, k, cv2.MORPH_DILATE, 1)
    reconMarkerMask = bZ.otsuBinarize(iO.andOperation(reconMarker, mask),1)
    reconMarkerMask = iO.applyTransformation(reconMarkerMask, k, cv2.MORPH_DILATE, 1)
    bloodMasked = marker.copy()
    bloodMasked = bloodMasked[:, :, 0]
    bloodMasked = iO.andOperation(bloodMasked, reconMarkerMask)
    laplacian = cv2.Laplacian(bloodMasked,cv2.CV_64F)
    laplacian[laplacian[:,:] > 60] = 0
    lapBina = bZ.simpleBinarization(laplacian, 1)
    aglutinQuad = inA2.imageQuadrantSplit(lapBina)
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
        distancesBetweenPoints += sum(inA2.getDistanceBetweenPoints(qCoords))
        _, cnts, _ = cv2.findContours(
            q, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
            if(cv2.contourArea(c) > 2):
                areas += cv2.contourArea(c)
                distances += inA2.getContourDistance(c, nQ)
#        clusterBlobs = inA.blobDetect(clusteredMarker)
#        clusterArea = inA.areaEstimation(clusterBlobs)
    # DIAGNOSTIC DETERMINATION
    # Areas v2 (clustering)
#        clusterDiagnostic = inA.areaAnalysisAreasV2(
#            clusteredMarker, clusterBlobs, clusterArea, i)
    # Cnn
    with tf.Session(graph=graph) as sess:
        feed_dict_oldProcessing = {x: x_batch, pkeep: pkeepConst}
        result = sess.run(y, feed_dict=feed_dict_oldProcessing)
        if (result.item(0) > 0.75):
            cnnDiagnostic = 'P'
        else: 
            cnnDiagnostic = 'N'
    # Areas v1 (old areas)
#        oldDiagnostic = inA.areaAnalysis(
#            oldProcessBinMask, oldBlobs, oldArea, i)
    # Other models
    features = list(inA2.extractFeatures(
        clusteredMarker, features2Extract).values())
    features.append(noise)
    features.append(distances)
    features.append(distancesBetweenPoints)
    scaledFeatures = []
    for feature, mean, v in zip(features, means, variances):
        z = (feature - mean)/ math.sqrt(v)
        scaledFeatures.append(z)
    otherModelsDiagnostics = []
    for model, modelName in zip(models, fileNames):
        vals = np.array(scaledFeatures).reshape(1, -1)
        modelDiagnostic = 'P' if model.predict(vals)[0] > CNN_PROB_TH else 'N'
        otherModelsDiagnostics.append(modelDiagnostic)
    # DIAGNOSTIC SHOW
    sP.showImage(marker, figSize=(3, 3))
    print( model.predict(vals)[0])
    print(f'CNN diag: {cnnDiagnostic}')
    [print(f'Model {model} diag: {diagnostic}')
    for model, diagnostic in zip(fileNames, otherModelsDiagnostics)]
    if diag == cnnDiagnostic:
        if diag == 'P' or diag == 'p':
                tPcNN += 1
        else:
                tNcNN += 1
    else:
        if diag == 'P' or diag == 'p':
            fPcNN += 1
        else:
            fNcNN += 1
    if diag == modelDiagnostic:
        if diag == 'P' or diag == 'p':
                tPaNN += 1
        else:
                tNaNN += 1
    else:
        if diag == 'P' or diag == 'p':
             fPaNN += 1
        else:
            fNaNN += 1
markers.close()
#%% Confusion matrixes
cmCnn = np.array([[tPcNN, fNcNN], [fPcNN, tNcNN]])
truePositivesCnn = cmCnn[0, 0]
falsePositivesCnn = cmCnn[1, 0]
trueNegativesCnn = cmCnn[1, 1]
falseNegativesCnn = cmCnn[0, 1]
sensitivityCnn = truePositivesCnn/(truePositivesCnn+falseNegativesCnn)
specificityCnn = trueNegativesCnn/(trueNegativesCnn+falsePositivesCnn)
cmAnn = np.array([[tPaNN, fNaNN], [fPaNN, tNaNN]])
truePositivesAnn = cmAnn[0, 0]
falsePositivesAnn = cmAnn[1, 0]
trueNegativesAnn = cmAnn[1, 1]
falseNegativesAnn = cmAnn[0, 1]
sensitivityAnn = truePositivesAnn/(truePositivesAnn+falseNegativesAnn)
specificityAnn = trueNegativesAnn/(trueNegativesAnn+falsePositivesAnn)