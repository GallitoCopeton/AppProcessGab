# %%
import math
import json

import numpy as np

import qrQuery
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from machineLearningUtilities import modelPerformance  as moPe
from IF2.Shows.showProcesses import showImage as sI
from IF2.ReadImage import readImage as rI

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
#%% Validation database
zeptoConnection = connections['zepto']
zaptoImagesCollection = qrQuery.getCollection(
    zeptoConnection['URI'], zeptoConnection['databaseName'], zeptoConnection['collections']['markersCollectionName'])
#%% Model loading
allModelsFolder = '../Models/ANNs'
modelFolders = ['ANN_0.85 date Jan 23 16_08_21', 'ANN_0.81 date Jan 24 15_48_32', 'ANN_0.81 date Jan 24 15_45_39', 'ANN_0.8 date Jan 24 16_16_38']
modelPaths = ['/'.join([allModelsFolder, folder, folder+'.pkl']) for folder in modelFolders]
models = [moPe.loadModel(path) for path in modelPaths]
#%% Model info loading
infoPaths = ['/'.join([allModelsFolder, folder, 'nnInfo.json']) for folder in modelFolders]
modelsInfo = [moPe.loadModelInfo(path) for path in infoPaths]
#%% Query and data fix
query = {'diagnostic': {'$ne': None}}
limit = 0
markers = zaptoImagesCollection.find(query).limit(limit)
markersInfo = [[(iO.resizeFixed(rI.readb64(marker['image']))),
                {'diagnostic': marker['diagnostic'],
                 'name':  marker['marker'],
                 'qr': marker['QR'],
                 'count': marker['count']}
                ] for marker in markers]
markerImages = [info[0] for info in markersInfo]
markersInfo = [info[1] for info in markersInfo]
#%% Model validation
modelPerformance = {}
for i, (markerImage, markerInfo) in enumerate(zip(markerImages, markersInfo)):
    print('*'*80)
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
#        print(f'El resultado del modelo {modelName} es {model.predict(vals)}')
        if not modelName in modelPerformance.keys():
            modelPerformance[modelName] = {}
            modelPerformance[modelName]['tP'] = 0
            modelPerformance[modelName]['tN'] = 0
            modelPerformance[modelName]['fP'] = 0
            modelPerformance[modelName]['fN'] = 0
        modelPred = 1 if model.predict(vals)[0][0] > .70 else 0
        print(f'REAL DIAGNOSTIC: {diagnostic}')
        print(f'Model {modelName} prediction: {modelPred}')
        if modelPred == diagnostic:
            if diagnostic == 1:
                modelPerformance[modelName]['tP'] += 1
            else:
                modelPerformance[modelName]['tN'] += 1
        else:
            if diagnostic == 1:
                modelPerformance[modelName]['fN'] += 1
                 
            else:
                modelPerformance[modelName]['fP'] += 1
for modelName in modelFolders:
    modelPerformance[modelName]['accuracy'] = (modelPerformance[modelName]['tP'] + modelPerformance[modelName]['tN']) / len(markersInfo)
    modelPerformance[modelName]['recall'] = modelPerformance[modelName]['tP'] / (modelPerformance[modelName]['tP'] + modelPerformance[modelName]['fN'])
    modelPerformance[modelName]['precision'] = modelPerformance[modelName]['tP'] / (modelPerformance[modelName]['tP'] + modelPerformance[modelName]['fP'])
    modelPerformance[modelName]['f1'] = 2 * ((modelPerformance[modelName]['precision'] * modelPerformance[modelName]['recall']) / (modelPerformance[modelName]['precision'] + modelPerformance[modelName]['recall']))


#        print(modelPerformance)
#    sI(markerImage, title=diagnostic)
        