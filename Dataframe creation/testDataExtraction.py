# %%
import datetime
import re
import json

import pandas as pd
import numpy as np

import qrQuery
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.ReadImage import readImage as rI

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
xploraStaging = connections['stagingXplora']
xploraStagingImagesCollection = qrQuery.getCollection(
xploraStaging['URI'], xploraStaging['databaseName'], xploraStaging['collections']['imagesCollectionName'])
xploraStagingDataCollection = qrQuery.getCollection(
xploraStaging['URI'], xploraStaging['databaseName'], xploraStaging['collections']['dataCollectionName'])
#%% Get the requested qrs ranges
requestedQrs = [('102191200101178', '102191200101193')]
qrList = []
for reqRange in requestedQrs:
    initial = int(reqRange[0])
    last = int(reqRange[1])
    fullList = np.arange(initial, last+1, 1)
    fullListStr = [str(qr) for qr in fullList]
    qrList += fullListStr
#%%
query = {'qrCode': {'$in': qrList}}
documents = xploraStagingDataCollection.find(query)
fullFeatures = []
for i, document in enumerate(documents):
    # Info extraction
    columns = ['qr', 'count']
    qr = document['qrCode']
    count = document['count']
    markers = document['marker']
    markerNames = []
    a1Name = []
    a2Name = []
    totalMarkerRes = []
    for marker in markers:
        totalMarkerRes.append(marker['analysisDetails'][1]['value'])
        totalMarkerRes.append(marker['analysisDetails'][-1]['value'])
        markerNames.append(marker['name'])
        a1Name.append(marker['analysisDetails'][1]['algorithm'])
        a2Name.append(marker['analysisDetails'][-1]['algorithm'])
    markerFeatureNames = list(set(a1Name + a2Name))
    iterables = [markerNames, markerFeatureNames]
    index = pd.MultiIndex.from_product(iterables, names=['name', 'data'])
    p = pd.DataFrame(np.array(totalMarkerRes).reshape(1, len(totalMarkerRes)),columns=index)
    p2 = pd.DataFrame(np.array([qr, count]).reshape(1, len(columns)), columns=columns)
    p3 = p2.join(p, how='inner')
    fullFeatures.append(p3)
pConc = pd.concat(fullFeatures, axis=0)
pConc.to_excel('resultados2.xlsx', index=False)
#fullDataframe = pd.DataFrame(fullFeatures, columns=features2Extract)