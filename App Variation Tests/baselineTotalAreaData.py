# %%
import datetime
import re
import json

import pandas as pd
import numpy as np
import seaborn as sns

import qrQuery
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.ReadImage import readImage as rI
from IF2.Crop import croppingProcess as cP
from IF2.Shows.showProcesses import showImage as show

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
zaptoConnection = connections['zapto']
zaptoImagesCollection = qrQuery.getCollection(
    zaptoConnection['URI'], zaptoConnection['databaseName'], zaptoConnection['collections']['markersCollectionName'])
#%% 
limit = int(3800)
query = [
        {'$match': {'diagnostic': {'$ne': None}}},
        {'$sample': {'size': limit}}
         ]
markers = zaptoImagesCollection.aggregate(query, allowDiskUse=True)
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
#%%
features2Extract = ['totalArea', 'noise', 'diagnostic']
fullFeatures = []
for i, (marker, info) in enumerate(zip(markerImages[:5000], markersInfo[:5000])):
    print(f'\nProcesando marcador {i+1} de {registerCount}')
    # Info extraction
    name = info['name']
    qr = info['qr']
    count = info['count']
    diagnostic = inA.fixDiagnostic(info['diagnostic'])
    print(diagnostic)
    features = inA.extractFeatures(marker, features2Extract)
    featureListNames = sorted(
        features.keys(), key=lambda i: features2Extract.index(i))
    featureList = [features[name] for name in featureListNames]
    featureList.append(diagnostic)
    fullFeatures.append(featureList)
fullDataframe = pd.DataFrame(fullFeatures, columns=features2Extract)
#%% Cleaning and preparing data
fullDataframe.dropna(inplace=True)
neg = fullDataframe[fullDataframe['diagnostic'] == 0].astype(int)
pos = fullDataframe[fullDataframe['diagnostic'] == 1].astype(int)
sns.distplot(neg['noise'], bins=10, kde=False)
sns.distplot(pos['noise'], bins=10, kde=False)
posDes = pos.describe()
negDes = neg.describe()