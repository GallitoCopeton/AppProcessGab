# %%
import datetime
import re
import json

import pandas as pd

import qrQuery
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.ReadImage import readImage as rI

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
# %%
zaptoConnection = connections['zapto']
zaptoImagesCollection = qrQuery.getCollection(
    zaptoConnection['URI'], zaptoConnection['databaseName'], zaptoConnection['collections']['markersCollectionName'])
# %%
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
# %%
features2Extract = ['totalArea',
                    'fullBlobs', 'bigBlobs', 'medBlobs',
                    'noise',
                    'distanceBetweenPoints',
                    'diagnostic']
registerCount = len(markersInfo)
fullFeatures = []
for i, (marker, info) in enumerate(zip(markerImages, markersInfo)):
    print(f'\nProcesando marcador {i+1} de {registerCount}')
    # Info extraction
    name = info['name']
    qr = info['qr']
    count = info['count']
    diagnostic = inA.fixDiagnostic(info['diagnostic'])
    features = inA.extractFeatures(marker, features2Extract)
    featureListNames = sorted(
        features.keys(), key=lambda i: features2Extract.index(i))
    featureList = [features[name] for name in featureListNames]
    featureList.append(diagnostic)
    fullFeatures.append(featureList)
fullDataframe = pd.DataFrame(fullFeatures, columns=features2Extract)
# %% Save it
todaysDate = datetime.datetime.now()
dateString = re.sub(r':', '_', todaysDate.ctime())[4:-5]
dataframesFolder = '../Feature Tables'
currentDfFolder = f'DF {dateString}'
dfFilename = currentDfFolder + '.xlsx'
currentDfPath = '/'.join([dataframesFolder, currentDfFolder])
qrQuery.makeFolders(currentDfPath)
dfFilePath = '/'.join([currentDfPath, dfFilename])
fullDataframe.to_excel(dfFilePath, index=False)
# Info txt save
dfInfoFileName = 'dfInfo.txt'
dfInfoFilePath = '/'.join([currentDfPath, dfInfoFileName])
joinedFeatures = ', '.join(fullDataframe.columns)
nRows = str(len(fullDataframe))
with open(dfInfoFilePath, 'w') as infoFile:
    infoFile.write(
        f'Number of tests included: {nRows}\nFeatures used: {joinedFeatures}')