# %%
import datetime
import os
import math
import json

import numpy as np

import qrQuery
from AppProcess.MarkerProcess import markerProcess as mP
from ImageProcessing import blobAnalysis as bA
from ImageProcessing import colorStats as cS
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA
from machineLearningUtilities import nnUtils as nnU
from machineLearningUtilities import modelPerformance  as moPe
from ShowProcess.showProcesses import showImage as sI
from ReadImages import readImage as rI

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
#%% Validation database
zeptoConnection = connections['zapto']
zaptoImagesCollection = qrQuery.getCollection(
    zeptoConnection['URI'], zeptoConnection['databaseName'], zeptoConnection['collections']['markersCollectionName'])
#%% Query and data fix
query = {'diagnostic': {'$ne': None}}
limit = 10
markers = zaptoImagesCollection.find(query).limit(limit)
markersInfo = [[(iO.resizeFixed(rI.readb64(marker['image']))),
                {'diagnostic': marker['diagnostic'],
                 'name':  marker['marker'],
                 'qr': marker['QR'],
                 'count': marker['count']}
                ] for marker in markers]
markerImages = [info[0] for info in markersInfo]
markersInfo = [info[1] for info in markersInfo]
#%% Model loading
allModelsFolder = '../Models/ANNs'
modelFolders = ['ANN_0.85 date Jan 23 16_08_21']
modelPaths = ['/'.join([allModelsFolder, folder, folder+'.pkl']) for folder in modelFolders]
models = [moPe.loadModel(path) for path in modelPaths]
#%% Model info loading
infoPaths = ['/'.join([allModelsFolder, folder, 'nnInfo.json']) for folder in modelFolders]
modelsInfo = [moPe.loadModelInfo(path) for path in infoPaths]
#%% Model validation
for i, (markerImage, markerInfo) in enumerate(zip(markerImages, markersInfo)):
    diagnostic = inA.fixDiagnostic(markerInfo['diagnostic'])
    for model, info, modelName in zip(models, modelsInfo, modelFolders):
        features2Extract = info['features']
        features = inA.extractFeatures(markerImage, features2Extract)
        featureListNames = sorted(
            features.keys(), key=lambda i: features2Extract.index(i))
        featureList = [features[name] for name in featureListNames]
        scaledFeatures = []
        for feature, mean, v in zip(featureList, info['means'], info['variances']):
            z = (feature - mean)/ math.sqrt(v)
            scaledFeatures.append(z)
        vals = np.array(scaledFeatures).reshape(1, -1)
        print(model.predict(vals))
    sI(markerImage, title=diagnostic)
        