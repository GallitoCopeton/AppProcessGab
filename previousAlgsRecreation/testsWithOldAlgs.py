# %%
import base64
import datetime
import operator
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
import pymongo as pymongo
import tensorflow as tf
from matplotlib import pyplot as plt

import indAnalysis as inA
import perspective as pPe
import preProcessing as pP
import qrQuery
import readImage
import sorts as srt
from AppProcess.MarkerProcess import markerProcess
from ImageProcessing import binarizations as bZ
from ImageProcessing import blobAnalysis as bA
from ImageProcessing import colorTransformations as cT
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA2
from machineLearningUtilities import modelPerformance as mP
from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP


def fixMarker(marker): return iO.resizeFixed(rI.readb64(marker['image']))


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# Scripts para leer y procesar imagen
sys.path.insert(0, '../Golden Master (AS IS)')


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
# %%
# Info of the markers we want to analyze
markerNamesReal = ['RV1681', 'ESAT6', 'P24', 'CFP10']
features2Extract = ['nBlobs', 'totalArea', 'fullBlobs', 'bigBlobs', 'medBlobs',
                    'smallBlobs', 'q0HasBlob', 'q1HasBlob', 'q2HasBlob', 'q3HasBlob', 'diagnostic']
# Query: markers I want, that their diagnostic exists
limit = 20
mask = inA.readMask()
markers = realMarkerCollection.find(
    {'marker': {'$in': markerNamesReal}, 'diagnostic': {'$ne': None}}, no_cursor_timeout=True).limit(limit).sort('_id', -1)
blobMarkers = []
listoldProcesssBinary = []


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


pkeepConst = 1
# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/x:0')
y = graph.get_tensor_by_name('prefix/y_pred:0')
pkeep = graph.get_tensor_by_name('prefix/pkeep:0')
AREAS_DIAM_TH = 20
CNN_PROB_TH = 0.7
tP = 0
tN = 0
fP = 0
fN = 0
# %%
modelsFolder = '../Machine Learning Models'
fileNames = ['81.93% ANN.pkl', '94.88% ANN.pkl']
models = [mP.loadModel(modelsFolder, fileName) for fileName in fileNames]
for i, marker in enumerate(markers):
    print('*'*60)
    name = marker['marker']
    count = marker['count']
    diag = marker['diagnostic']
    print(f'Real diag: {diag}')
    marker = fixMarker(marker)
    # OLD IMAGE PROCESSING
    oldProcessBin = pP.contourBinarizationOtsu(
        marker, 3, 3, 45, 4, Gs=0, inverse=False, mean=True)
    oldProcessBinMask = inA.andOperation(oldProcessBin, mask)
    oldProcess = inA.erosionDilation(oldProcessBinMask, 3)
    oldProcess = cv2.bitwise_not(oldProcess)
    oldBlobs = inA.blobDetect(oldProcess)
    oldArea = inA.areaEstimation(oldBlobs)
    # CNN PROCESSING
    marker32 = marker.astype('float32')
    marker32 = np.multiply(marker32, 1.0/255.0)
    x_batch = marker32.reshape(1, 90, 90, 3)
    # CLUSTER PROCESSING
    clusteredMarker = markerProcess.clusteringProcess([marker], 3, 6)[0]
    clusterBlobs = inA.blobDetect(clusteredMarker)
    clusterArea = inA.areaEstimation(clusterBlobs)
    # DIAGNOSTIC DETERMINATION
    # Areas v2 (clustering)
    clusterDiagnostic = inA.areaAnalysisAreasV22(
        clusteredMarker, clusterBlobs, clusterArea, name)
    # Cnn
    with tf.Session(graph=graph) as sess:
        feed_dict_oldProcessing = {x: x_batch, pkeep: pkeepConst}
        result = sess.run(y, feed_dict=feed_dict_oldProcessing)
        if (result.item(0) > 0.75):
            cnnDiagnostic = 'P'
        if (result.item(1) > 0.75):
            cnnDiagnostic = 'N'
    # Areas v1 (old areas)
    oldDiagnostic = inA.areaAnalysis2(
        oldProcessBinMask, oldBlobs, oldArea, name)
    # Other models
    features = list(inA2.extractFeatures(clusteredMarker, features2Extract).values())
    otherModelsDiagnostics = []
    for model, modelName in zip(models, fileNames):
        vals = np.array(features).reshape(1, -1)
        modelDiagnostic = 'P' if model.predict(vals)[0] > CNN_PROB_TH else 'N'
        otherModelsDiagnostics.append(modelDiagnostic)
    
    # DIAGNOSTIC SHOW
    sP.showImage(marker, figSize=(3,3))    
    print(f'CNN diag: {cnnDiagnostic}')
    print(f'Cluster diag: {clusterDiagnostic}')
    print(f'Old blobs diag: {oldDiagnostic}')
    [print(f'Model {model} diag: {diagnostic}')
     for model, diagnostic in zip(fileNames, otherModelsDiagnostics)]
    # if result.item(0) < CNN_PROB_TH or oldArea < AREAS_DIAM_TH:
    #     defRes = 'areas'
    # else:
    #     defRes = 'red'
    # print(f'App would have chosen: {defRes}')
    # if diag == cnnDiagnostic:
    #     if diag == 'P' or diag == 'p':
    #         tP += 1
    #     else:
    #         tN += 1
    # else:
    #     if diag == 'P' or diag == 'p':
    #         fP += 1
    #     else:
    #         fN += 1
    #     pass
    print('*'*60)
markers.close()
