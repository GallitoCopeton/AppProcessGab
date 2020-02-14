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

limit = int(3168*.80)
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
# %%
features2Extract = [
#                    'noise',
#                    'roc',
                    'agl',
                    'aglMean',
                    'aglDist',
#                    'distanceBetweenPoints',
#                    'distance',
                    'nBlobs',
#                    'value',
                    'totalArea',
#                    'anomaliesSize',
#                    'anomaliesDistance',
                    'fullBlobs',
                    'bigBlobs',
                    'medBlobs',
                    'smallBlobs',
#                    'q0HasBlob',
#                    'q1HasBlob',
#                    'q2HasBlob',
#                    'q3HasBlob',
#                    'q0Perimeter',
#                    'q1Perimeter',
#                    'q2Perimeter',
#                    'q3Perimeter',
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
    print(diagnostic)
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
dfInfoFileName = 'dfInfo.json'
dfInfoFilePath = '/'.join([currentDfPath, dfInfoFileName])
joinedFeatures = ','.join(fullDataframe.columns)
nRows = str(len(fullDataframe))
joinedIds = ','.join([str(markerInfo['_id']) for markerInfo in markersInfo])
jsonOutDict = {
        'features': joinedFeatures,
        'nTests': nRows,
        '_idsUsed': joinedIds
    }
with open(dfInfoFilePath, 'w') as jsonOut:
    json.dump(jsonOutDict, jsonOut)