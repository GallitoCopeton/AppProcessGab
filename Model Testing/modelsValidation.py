# %%
import math
import json
import os

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
zeptoConnection = connections['zapto']
zaptoImagesCollection = qrQuery.getCollection(
    zeptoConnection['URI'], zeptoConnection['databaseName'], zeptoConnection['collections']['markersCollectionName'])
#%% Model loading
allModelsFolder = '../Models/ANNs'
modelFolders = ['ANN_date Jan 28 08_32_24', 'ANN_date Jan 27 16_35_46', 'ANN_date Jan 28 11_10_58', 'ANN_date Jan 28 12_27_27']
modelPaths = ['/'.join([allModelsFolder, folder]) for folder in modelFolders]
modelsByPath = []
modelByPathNames = []
for modelPath in modelPaths:
    modelNames = os.listdir(modelPath)
    models = []
    names = []
    for model in modelNames:
        if model.endswith('.pkl'):
            modelFullPath = '/'.join([modelPath, model])
            names.append(model)
            model = moPe.loadModel(modelFullPath)
            models.append(model)
    modelByPathNames.append(names)
    modelsByPath.append(models)

#models = [moPe.loadModel(path) for path in modelPaths]
#%% Model info loading
infoPaths = ['/'.join([allModelsFolder, folder, 'nnInfo.json']) for folder in modelFolders]
modelsInfo = [moPe.loadModelInfo(path)['0'] for path in infoPaths]
#%% Query and data fix
query = {'diagnostic': {'$ne': None}}
limit = 2000
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
    for modelsInPath, modelNamesInPath, modelsInPathInfo in zip(modelsByPath, modelByPathNames, modelsInfo):
        features2Extract = modelsInPathInfo['features']
        features = inA.extractFeatures(markerImage, features2Extract)
        featureListNames = sorted(
            features.keys(), key=lambda i: features2Extract.index(i))
        featureList = [features[featureName] for featureName in featureListNames]
        scaledFeatures = []
        for feature, mean, v in zip(featureList, modelsInPathInfo['means'], modelsInPathInfo['variances']):
            z = (feature - mean)/ math.sqrt(v)
            scaledFeatures.append(z)
        for model, name in zip(modelsInPath, modelNamesInPath):
            vals = np.array(scaledFeatures).reshape(1, -1)
            if not name in modelPerformance.keys():
                modelPerformance[name] = {}
                modelPerformance[name]['tP'] = 0
                modelPerformance[name]['tN'] = 0
                modelPerformance[name]['fP'] = 0
                modelPerformance[name]['fN'] = 0
            modelPred = 1 if model.predict(vals)[0][0] > .5 else 0
            print(f'El resultado del modelo {name} es {modelPred}')
            if modelPred == diagnostic:
                if diagnostic == 1:
                    modelPerformance[name]['tP'] += 1
                else:
                    modelPerformance[name]['tN'] += 1
            else:
                if diagnostic == 1:
                    modelPerformance[name]['fN'] += 1
                     
                else:
                    modelPerformance[name]['fP'] += 1
    sI(markerImage, title=f'REAL DIAGNOSTIC: {diagnostic}')
for key in modelPerformance.keys():
    modelPerformance[key]['accuracy'] = (modelPerformance[key]['tP'] + modelPerformance[key]['tN']) / len(markersInfo)
    modelPerformance[key]['recall'] = modelPerformance[key]['tP'] / (modelPerformance[key]['tP'] + modelPerformance[key]['fN'])
    modelPerformance[key]['precision'] = modelPerformance[key]['tP'] / (modelPerformance[key]['tP'] + modelPerformance[key]['fP'])
    modelPerformance[key]['f1'] = 2 * ((modelPerformance[key]['precision'] * modelPerformance[key]['recall']) / (modelPerformance[key]['precision'] + modelPerformance[key]['recall']))


#        print(modelPerformance)
#    sI(markerImage, title=diagnostic)
        