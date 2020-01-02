#%%
import cv2
import pandas as pd
import numpy as np

import qrQuery
import datetime
from ReadImages import readImage as rI
from AppProcess.CroppingProcess import croppingProcess as cP
from AppProcess.MarkerProcess import markerProcess as mP
from ImageProcessing import colorTransformations as cT
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA
import ShowProcess.showProcesses as sP
#%%
URI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
dbName = 'findValidation'
collectionRegistersName = 'registerstotals'
collectionImagesName = 'imagestotals'
imagesCollection = qrQuery.getCollection(URI, dbName, collectionImagesName)
dataCollection = qrQuery.getCollection(URI, dbName, collectionRegistersName)
todaysDate = datetime.datetime.now()
startDay = 0
finishDay = 1
startDate = todaysDate - datetime.timedelta(days=startDay)
finishDate = startDate - datetime.timedelta(days=finishDay-startDay)
dateQuery = {'createdAt': {
    '$lt': startDate, '$gte': finishDate
}}
#%%
dataRecords = dataCollection.find({}, limit=10).sort('_id', -1)
markerNames = ['ESAT6', 'CFP10', 'RV1681', 'CTR']
iterables = [markerNames, ['nBlobs', 'value', 'totalArea', 'count','diagnostic']]
index = pd.MultiIndex.from_product(iterables, names=['Protein', 'Info'])
kColors = 3
attempts = 2
fullDf = []
for i, document in enumerate(dataRecords):
    qr = document['qrCode']
    count = document['count']
    device = document['macaddress']
#    print(f'Image {i+1} QR {qr} count {count} device {device}')
    imageQuery = {'fileName': document['qrCode'], 'count': document['count']}
    imagesCount = qrQuery.getDocumentCount(imagesCollection, imageQuery)
    if imagesCount == 0:

#        print('No existe imagen')
        continue
    image = cT.BGR2RGB(rI.customQuery(imagesCollection, imageQuery)[0]['file'])
    try:
        testSite = cP.getNonEqTestSite(image)
    except Exception as e:
        print(e)
    try:
        markers = cP.getMarkers(testSite)
    except Exception as e:
        print(e)
    markersNot = mP.clusteringProcess(
        markers, kColors, attempts, extendedProcess=False)
    markersInfo = []
    dfs = []
    try:
        for markerRegister, marker in zip(document['marker'], markersNot):
            nBlobs = markerRegister['analysisDetails'][2]['blobQty']
            value = markerRegister['analysisDetails'][2]['value']
            diagnostic = markerRegister['analysisDetails'][2]['diagnostic']
            totalArea = 0
            try:
                totalArea = markerRegister['analysisDetails'][2]['totalArea']
            except:
                quadrants = inA.imageQuadrantSplit(iO.notOperation(marker))
                for nQuadrant, q in enumerate(quadrants):
                    sideOfImage = q[0].shape[0]
                    _, contours, _ = cv2.findContours(
                        q, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0 and len(contours[0]) >= 4:
                        for nContour, contour in enumerate(contours):
                            if inA.blobValidation(contour, nQuadrant, sideOfImage):
                                totalArea += cv2.contourArea(contour)/490 * 100
            markersInfo += [nBlobs, value, totalArea, count, diagnostic]
            if len(markersInfo) != 6:
                continue
        df = pd.DataFrame(np.array(markersInfo).reshape(1, len(markersInfo)), columns=index)
    except Exception as e:
        print(e)
        continue
    df['qrCode'] = qr
    df['device'] = device
    cols = list(df)
    # move the column to head of list using index, pop and insert
    cols.insert(0, cols.pop(cols.index(('qrCode', ''))))
    # use ix to reorder
    df = df.loc[:, cols]
    fullDf.append(df)
#    try:
#        mergedMarkers = inA.mergeQuadrants(markersNot)
    sP.showImage(testSite, figSize=(6,6))
#        sP.showImage(mergedMarkers, figSize=(4,4))
#    except:
#        continue
fullDf = pd.concat(fullDf)
fullDf.set_index('qrCode', inplace=True)
#%%
fullDf.to_excel('20 registers.xlsx')





