# %%
import base64
import datetime
import operator
import os
import re
import sys
import math

import cv2
import numpy as np
import pandas as pd
import pymongo
import tensorflow as tf
from matplotlib import pyplot as plt
#sys.path.insert(0, '../Golden Master (AS IS)')
#import indAnalysis as inA
#import preProcessing as pP
from sklearn.preprocessing import StandardScaler

import qrQuery
from AppProcess.MarkerProcess import markerProcess
from AppProcess.CroppingProcess import croppingProcess as cP
from ImageProcessing import binarizations as bZ
from ImageProcessing import colorTransformations as cT
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA2
from machineLearningUtilities import modelPerformance as mP
from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import nnUtils as nnU


def fixImage(image): return iO.resizeImg(rI.readb64(image['file']), 728)


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
modelsFolder = './'
fileNames = ['ANN_89.pkl']
models = [mP.loadModel(modelsFolder, fileName) for fileName in fileNames]

# %% Collections
zeptoURI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
zeptoDbName = 'findValidation'
zeptoCollectionRegistersName = 'registerstotals'
zeptoCollectionImagesName = 'imagestotals'
zeptoImagesCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, zeptoCollectionRegistersName)
zeptoImagesCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, zeptoCollectionImagesName)

realURI = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
realDbName = 'prodLaboratorio'
realCollectionName = 'markerTotals'
realMarkerCollection = qrQuery.getCollection(
    realURI, realDbName, realCollectionName)

kColors = 3
attempts = 6
# %%
# Info of the markers we want to analyze
markerNamesReal = ['RV1681', 'ESAT6', 'P24', 'CFP10']
features2Extract = ['nBlobs', 'totalArea', 'fullBlobs', 'bigBlobs', 'medBlobs', 'smallBlobs', 'q0HasBlob', 'q1HasBlob', 'q2HasBlob', 'q3HasBlob', 'noise', 'distance', 'diagnostic']
# Query: markers I want, that their diagnostic exists
limit = 100
#mask = inA.readMask()
images = zeptoImagesCollection.find({}).limit(limit).sort('_id', 1)
# CNN initialization
graph = load_graph('../data/opt_unimaHealth.pb')
x = graph.get_tensor_by_name('prefix/x:0')
y = graph.get_tensor_by_name('prefix/y_pred:0')
pkeep = graph.get_tensor_by_name('prefix/pkeep:0')
pkeepConst = 1
AREAS_DIAM_TH = 20
CNN_PROB_TH = 0.7
# Info of the markers we want to analyze
mask = bZ.otsuBinarize(cT.BGR2gray(
    rI.readLocal('../Smoothness quantifying/mask.png')))
mask = iO.applyTransformation(mask, np.ones((3, 3)), cv2.MORPH_ERODE, 2)
k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
kernelE = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
localImagesPath = '../Zepto/Negativas'
#images = [iO.resizeImg(rI.readLocal('/'.join([localImagesPath, name])), 728) for name in os.listdir(localImagesPath) if name.endswith('.jpg')]
#%%
sameP = 0
difP = 0
sameN = 0
difN = 0
for i, image in enumerate(images):
    print('*'*60)
    image = cT.BGR2RGB(fixImage(image))
    try:
        testSite = cP.getTestArea(image)
    except Exception as e:
        pass
    try:
        markers = cP.getMarkers(testSite)
    except Exception as e:
        pass
    for i, marker in enumerate(markers):
        # OLD IMAGE PROCESSING
#        oldProcessBin = pP.contourBinarizationOtsu(
#            marker, 3, 3, 45, 4, Gs=0, inverse=False, mean=True)
#        oldProcessBinMask = inA.andOperation(oldProcessBin, mask)
#        oldProcess = inA.erosionDilation(oldProcessBinMask, 3)
#        oldProcess = cv2.bitwise_not(oldProcess)
#        oldBlobs = inA.blobDetect(oldProcess)
#        oldArea = inA.areaEstimation(oldBlobs)
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
        
        for feature, mean, v in zip(features, means, var):
            z = (feature - mean)/ math.sqrt(v)
            scaledFeatures.append(z)
        otherModelsDiagnostics = []
        for model, modelName in zip(models, fileNames):
            vals = np.array(scaledFeatures).reshape(1, -1)
            modelDiagnostic = 'P' if model.predict(vals)[0] > CNN_PROB_TH else 'N'
            print(model.predict(vals))
            otherModelsDiagnostics.append(modelDiagnostic)
        
        
        if modelDiagnostic == cnnDiagnostic and cnnDiagnostic == 'P':
            sameP += 1
        elif modelDiagnostic != cnnDiagnostic and cnnDiagnostic == 'N':
            difP += 1
        if modelDiagnostic == cnnDiagnostic and cnnDiagnostic == 'N':
            sameN += 1
        elif modelDiagnostic != cnnDiagnostic and cnnDiagnostic == 'P':
            difN += 1
        print(sameP, difP, sameN, difN)
        # DIAGNOSTIC SHOW
        sP.showImage(marker, figSize=(3, 3))
        print(f'CNN diag: {cnnDiagnostic}')
#        print(f'Cluster diag: {clusterDiagnostic}')
#        print(f'Old blobs diag: {oldDiagnostic}')
        [print(f'Model {model} diag: {diagnostic}')
        for model, diagnostic in zip(fileNames, otherModelsDiagnostics)]
    print('*'*60)
#%%
featuresToExtract = ['nBlobs', 'totalArea', 'fullBlobs', 'bigBlobs', 'medBlobs', 'smallBlobs', 'q0HasBlob', 'q1HasBlob', 'q2HasBlob', 'q3HasBlob', 'noise', 'distance', 'diagnostic']
#featuresToExtract = ['noise', 'distance', 'distanceBetweenPoints', 'diagnostic']

graph = load_graph('../data/opt_unimaHealth.pb')
x = graph.get_tensor_by_name('prefix/x:0')
y = graph.get_tensor_by_name('prefix/y_pred:0')
pkeep = graph.get_tensor_by_name('prefix/pkeep:0')
pkeepConst = 1
def fixMarker(image): return iO.resizeFixed(rI.readb64(image['image']))
limit = 1180
markers = realMarkerCollection.find({'diagnostic': {'$ne': None}}).limit(limit).sort('_id', -1)
fullDf = []
for i, marker in enumerate(markers):
    print('_'*60)
    stds = 0
    name = marker['marker']
    diagnostic = marker['diagnostic']
    marker = fixMarker(marker)
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
        stds += dx + dx
        _, cnts, _ = cv2.findContours(
            q, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
            if(cv2.contourArea(c) > 2):
                areas += cv2.contourArea(c)
                distances += inA2.getContourDistance(c, nQ)
    # DIAGNOSTIC SHOW
#    sP.showImage(marker, figSize=(3, 3),title=diagnostic)
#    sP.showImage(lapBina, figSize=(3, 3), axisOff=False)
#    spreadRatio = (stds)
#    print(f'Spread/number of aglutinations ratio: {spreadRatio}')
#    print(f'CNN diag: {cnnDiagnostic}')
#    print(f'Cluster diag: {clusterDiagnostic}')
#    print(f'Old blobs diag: {oldDiagnostic}')
#    [print(f'Model {model} diag: {diagnostic}')
#    for model, diagnostic in zip(fileNames, otherModelsDiagnostics)]
    dfDict = {}
    features = []
    features = list(inA2.extractFeatures(clusteredMarker, featuresToExtract).values())
    features.append(noise)
    features.append(distances)
#    features.append(distancesBetweenPoints)
    diagnostic = 1 if diagnostic == 'P' or diagnostic == 'p' else 0
    features.append(diagnostic)
    fullDf.append(pd.DataFrame.from_records([features], columns=featuresToExtract))
    print('_'*60)
fullDf = pd.concat(fullDf)
# %% Train and test set
#df = pd.read_excel('../Feature tables/Dataframe de Jan 20 09_37_04.xlsx')
X = mlU.getFeatures(fullDf, 0, -1)
y = mlU.getLabels(fullDf, 'diagnostic')
# %% Split data
X_train, X_test, y_train, y_test = mlU.splitData(X, y, .2)
# %% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
means = sc.mean_
variances = sc.var_
# %% Baseline model
alpha = .8
nFeatures = X.shape[1]
outputNeurons = 1
nSamples = len(X_train)
activations = ['relu','relu', 'sigmoid']
l1 = 0.7
l2 = 0.7
dropout = 0.5
batchNorm = False
model = nnU.createANN(alpha=alpha, features=nFeatures, outputNeurons=outputNeurons, nSamples=nSamples, activations=activations, l1=l1, l2=l2, dropout=dropout, batchNorm=batchNorm)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
modelHistory = model.fit(X_train, y_train, batch_size=500, epochs=2500, verbose=2, validation_data=(X_test, y_test))
nnU.plot_history([('Base model', modelHistory)])
yPred = nnU.performance(model, X_test, y_test)
#%%
filePath = './'
fileName = 'ANN_92.pkl'

mP.saveModel(filePath, fileName, model)