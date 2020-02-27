# %%
import datetime
import json
import re
import os
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score

import machineLearningUtilities.dataPreparation as mlU
import qrQuery
from IF2.Crop import croppingProcess as cP
from IF2.Marker import markerProcess as maP
from IF2.Processing import colorTransformations as cT
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.Processing import preProcessing as pP
from IF2.ReadImage import readImage as rI
from IF2.Shows.showProcesses import showImage as show
from machineLearningUtilities import modelPerformance as mP
from machineLearningUtilities import nnUtils as nnU

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
zaptoConnection = connections['zapto']
zaptoImagesCollection = qrQuery.getCollection(
    zaptoConnection['URI'], zaptoConnection['databaseName'], zaptoConnection['collections']['markersCollectionName'])
# %%
markers = zaptoImagesCollection.find(
    {'diagnostic': {'$ne': None}}).sort('_id', 1)
markersInfo = [[(iO.resizeFixed(rI.readb64(marker['image']))),
                {'diagnostic': marker['diagnostic'],
                 'name':  marker['marker'],
                 'qr': marker['QR'],
                 'count': marker['count'],
                 '_id': marker['_id']}
                ] for marker in markers]
markerImages = [info[0] for info in markersInfo]
markersInfo = [info[1] for info in markersInfo]
registerCount = len(markersInfo)
# %%
fullFeatures = []
kernel = np.array([[-0, -2, -0], [-1, 7, -1], [-0, -2, -0]])
for i, (marker, info) in enumerate(zip(markerImages, markersInfo)):
    print(f'\nProcesando marcador {i+1} de {registerCount}')
    print(info['diagnostic'])
    # Info extraction
#    markerDenoised = cv2.filter2D(marker, -1, kernel)
#    markerGray = pP.adapHistogramEq(marker, 5, (5, 5))
#    show(markerGray)
    markerGray = cT.BGR2gray(maP.getBloodOnlyMask(marker))
    
    marker1d = iO.resizeFixed(markerGray, 45).reshape((-1, 1)).ravel()
    marker1d = list(marker1d)
    marker1d.append(inA.fixDiagnostic(info['diagnostic']))
    fullFeatures.append(marker1d)

# %%
fullFeatures = np.array(fullFeatures).ravel().reshape(
    len(markersInfo), len(marker1d))
#fullDataframe.to_excel('marker1d.xlsx', index=False)
# %%
print(fullFeatures.shape)
fullDataframe = pd.DataFrame(fullFeatures)
fullDataframe.dropna(inplace=True)
X = fullDataframe.iloc[:, :-1].values
#X.drop('totalArea', inplace=True, axis=1)
#X.drop('agl', inplace=True, axis=1)
#X.drop('fullBlobs', inplace=True, axis=1)
#X.drop('smallBlobs', inplace=True, axis=1)
y = fullDataframe.iloc[:, -1].values
y = y.astype('int')
X_train, X_test, y_train, y_test = mlU.splitData(X, y, seed=56)
#%%
mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
xMax = mm.data_max_
xMin = mm.data_min_
# %%
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=100000)
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
plt.figure(figsize=(11, 11))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5,
            square=True, cmap='Pastel1')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
print(X.shape[1])
#%%
#print('Best:', best_model.best_estimator_.get_params())
betterLR = logisticRegr
predictions = betterLR.predict(X_test)
score = logisticRegr.score(X_test, y_test)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
plt.figure(figsize=(11, 11))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5,
            square=True, cmap='Pastel1')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
#%%
kf = cross_val_score(logisticRegr, X_train, y_train, cv=3)
#%%
print(kf.std())
print(kf.mean())
#%% new preds
picturesPath = '../assetsForTests/neg_finger_blood/mid/'
picturesFullPath = [picturesPath+name for name in os.listdir(picturesPath) if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg') or name.endswith('.JPG')]
pictures = [iO.resizeImg(rI.readLocal(path), 728) for path in picturesFullPath]
#%%
markerNames = ['E6', 'CF', 'RV']
columns = ['fileName'] + markerNames
fullData = []
for i, (picture, name) in enumerate(zip(pictures, os.listdir(picturesPath))):
    testArea = cP.getTestArea(picture)
    markers = cP.getMarkers(testArea)[:-1]
    diags= []
    for marker in markers:
#        markerDenoised = cv2.filter2D(marker, -1, kernel)
        markerGray = pP.normalizeLight(marker)
        markerGray = cT.BGR2gray(maP.getBloodOnlyMask(marker))
        marker1d = iO.resizeFixed(markerGray, 45).reshape((1, -1))
#        marker1d = list(marker1d)
#        scaledFeatures = []
#        for feature, Max, Min in zip(marker1d, xMax, xMin):
#            z = (feature - Min)/(Max-Min)
#            scaledFeatures.append(z)
#        scaledFeatures = np.array(scaledFeatures).reshape(1, -1)
        diagProb = logisticRegr.predict_proba(marker1d)[0][1]
        diag = logisticRegr.predict(marker1d)[0]
        diags.append(diag)
    diagStrings = ''.join([f'Diagnóstico para {marker}: {round(diag, 5)}\n' for marker, diag in zip(markerNames, diags)])
    data = [name] + diags
    fullData.append(data)
    show(testArea, title=diagStrings)
testDf1 = pd.DataFrame(fullData, columns=columns)
#%% Validation database
zeptoConnection = connections['zepto']
zaptoImagesCollection = qrQuery.getCollection(
    zeptoConnection['URI'], zeptoConnection['databaseName'], zeptoConnection['collections']['markersCollectionName'])
limit = 500
markers = zaptoImagesCollection.find({'diagnostic': {'$ne': None}}).limit(limit)
markersInfo = [[(iO.resizeFixed(rI.readb64(marker['image']))),
                {'diagnostic': marker['diagnostic'],
                 'name':  marker['marker'],
                 'qr': marker['QR'],
                 'count': marker['count'],
                 '_id': marker['_id']}
                ] for marker in markers]
markerImages = [info[0] for info in markersInfo]
markersInfo = [info[1] for info in markersInfo]
for marker, info in zip(markerImages, markersInfo):
    markerDenoised = cv2.filter2D(marker, -1, kernel)
    markerGray = cT.BGR2gray(markerDenoised)
    marker1d = markerGray.reshape((1, -1))
    d = info['diagnostic']
    show(markerDenoised, title=f'Predicted {betterLR.predict(marker1d)[0]} real {d}')