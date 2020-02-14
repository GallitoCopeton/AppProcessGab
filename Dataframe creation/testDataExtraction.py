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
from IF2.Crop import croppingProcess as cP
from IF2.Shows.showProcesses import showImage as show

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
xploraStaging = connections['stagingXplora']
xploraStagingImagesCollection = qrQuery.getCollection(
xploraStaging['URI'], xploraStaging['databaseName'], xploraStaging['collections']['imagesCollectionName'])
xploraStagingDataCollection = qrQuery.getCollection(
xploraStaging['URI'], xploraStaging['databaseName'], xploraStaging['collections']['dataCollectionName'])
#%% Get the requested qrs ranges
requestedQrs = [('102191200101195', '102191200101243')]
qrList = []
for reqRange in requestedQrs:
    initial = int(reqRange[0])
    last = int(reqRange[1])
    fullList = np.arange(initial, last+1, 1)
    fullListStr = [str(qr) for qr in fullList]
    qrList += fullListStr
query = {'qrCode': {'$in': qrList}}
#%% Data
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
        totalMarkerRes.append(marker['analysisDetails'][1]['blobQty'])
        totalMarkerRes.append(marker['analysisDetails'][-1]['blobQty'])
        markerNames.append(marker['name'])
        a1Name.append('nBlobsav1')
        a1Name.append(marker['analysisDetails'][1]['algorithm'])
        
        a2Name.append(marker['analysisDetails'][-1]['algorithm'])
        a2Name.append('nBlobsav2')
        
    markerFeatureNames = list(set(a1Name + a2Name))
    iterables = [markerNames, markerFeatureNames]
    index = pd.MultiIndex.from_product(iterables, names=['name', 'data'])
    p = pd.DataFrame(np.array(totalMarkerRes).reshape(1, len(totalMarkerRes)),columns=index)
    p2 = pd.DataFrame(np.array([qr, count]).reshape(1, len(columns)), columns=columns)
    p3 = p2.join(p, how='inner')
    fullFeatures.append(p3)
pConc = pd.concat(fullFeatures, axis=0)
pConc.to_excel('resultadosComp.xlsx', index=False)
#fullDataframe = pd.DataFrame(fullFeatures, columns=features2Extract)
#%% Images# %%
query = {'fileName': {'$in': qrList}}
imageDocuments = xploraStagingImagesCollection.find(query)
imagesInfo = [[(iO.resizeImg(rI.readb64(image['file']))),
                {'qr': image['fileName'],
                 'count': image['count'],
                 '_id': image['_id']}
                ] for image in imageDocuments]
images = [info[0] for info in imagesInfo]
imagesInfo = [info[1] for info in imagesInfo]
markerNames = ['E6','CF','RV','CT']
features2Extract = ['agl',
                    'aglMean',
                    'totalArea',
                    'bigBlobs',
                    'medBlobs',
                    
                    ]
markerDfList = []
for picture, info in zip(images, imagesInfo):
    try:
        testArea = cP.getNonEqTestSite(picture)
        markers = cP.getMarkers(testArea)
    except:
        continue
    if len(markers) == 0: 
        continue
    allMarkersFeatures = []
    for marker, name in zip(markers, markerNames):
        features = inA.extractFeatures(marker, features2Extract)
        featureValues = list(features.values())
#        featureValues.append(int(info['qr']))
        allMarkersFeatures += featureValues
    iterables = [markerNames, features2Extract]
    index = pd.MultiIndex.from_product(iterables, names=['name', 'data'])
    singleMarkerDf = pd.DataFrame(np.array(allMarkersFeatures).reshape(1, len(allMarkersFeatures)),columns=index, index=[info['qr']])
    markerDfList.append(singleMarkerDf)
    show(testArea, figSize=(6,6), title=info['qr'])
completeMarkerDf = pd.concat(markerDfList)
corrE6 = completeMarkerDf['E6'].corr().T
corrCF = completeMarkerDf['CF'].corr().T
corrRV = completeMarkerDf['RV'].corr().T
corrCT = completeMarkerDf['CT'].corr().T
#%% 
#size = 1
#splitTestDfs = [completeMarkerDf.iloc[i:i+size,:] for i in range(0, len(completeMarkerDf),size)]
#customSplitDfs = []
qrs = [info['qr'] for info in imagesInfo]
#splitQrs = [qrs[i:i+size] for i in range(0, len(qrs),size)]
#for df, qr in zip(splitTestDfs, qrs):
#    descriptionDf = df.describe()
#    descriptionDf['qr'] = qr
#    customSplitDfs.append(descriptionDf)
#fullTestsDescriptions = pd.concat([df.describe().drop(['count','25%','50%','75%'], axis=0) for df in splitTestDfs])
#fullTestsDescriptions.to_excel('descriptions4.xlsx')
completeMarkerDf.to_excel('complete_measurements.xlsx')