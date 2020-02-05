import os
import json
from bson import ObjectId
import math

import numpy as np

from IF2.ReadImage import readImage as rI
from IF2.Processing import imageOperations as iO
from IF2.Crop import croppingProcess as cP
from IF2.Processing import indAnalysis as inA
from machineLearningUtilities import modelPerformance  as moPe
from IF2.Shows.showProcesses import showImage as show

picturesPath = '../assetsForTests/modelTesting/'
picturesFullPath = [picturesPath+name for name in os.listdir(picturesPath) if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg')]
pictures = [iO.resizeImg(rI.readLocal(path), 728) for path in picturesFullPath]

#%% Model loading
allModelsFolder = '../Models/ANNs'
modelFolders = ['ANN_date Feb  4 11_48_53']
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
#%% Model info loading
infoPaths = ['/'.join([allModelsFolder, folder, 'nnInfo.json']) for folder in modelFolders]
modelsInfo = [moPe.loadModelInfo(path)['0'] for path in infoPaths]
trainingFilesPaths = ['../Feature Tables/'+modelInfo['params']['trainingFileName']+'/dfInfo.json' for modelInfo in modelsInfo]
idsUsedByPath = []
for path in trainingFilesPaths:
    with open(path) as jsonFile:
        ids = [ObjectId(_id) for _id in json.load(jsonFile)['_idsUsed'].split(',')]
        idsUsedByPath += (ids)
idsUsedByPath = list(set(idsUsedByPath))
#%%
for picture in pictures:
    testArea = cP.getTestArea(picture)
    markers = cP.getMarkers(testArea)[:-1]
    for marker in markers:
        for modelsInPath, modelsInPathInfo in zip(modelsByPath, modelsInfo):
            features2Extract = modelsInPathInfo['features']
            features = inA.extractFeatures(marker, features2Extract)
            featureListNames = sorted(
                features.keys(), key=lambda i: features2Extract.index(i))
            featureList = [features[featureName] for featureName in featureListNames]
            scaledFeatures = []
            for feature, mean, v in zip(featureList, modelsInPathInfo['means'], modelsInPathInfo['variances']):
                z = (feature - mean)/ math.sqrt(v)
                scaledFeatures.append(z)
            for model in modelsInPath:
                vals = np.array(scaledFeatures).reshape(1, -1)
                modelPred = 1 if model.predict(vals)[0][0] > .7 else 0
                print(modelPred, model.predict(vals)[0][0])
    show(testArea, figSize=(7,7))