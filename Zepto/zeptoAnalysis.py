#%%
import cv2

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
todaysDate = datetime.datetime.now()
startDay = -1
finishDay = 1
startDate = todaysDate - datetime.timedelta(days=startDay)
finishDate = startDate - datetime.timedelta(days=finishDay-startDay)
dateQuery = {'createdAt': {
    '$lt': startDate, '$gte': finishDate
}}
images = rI.customQuery(imagesCollection, dateQuery, 10)
#%%
kColors = 3
attempts = 2
for image in images:
    image = cT.BGR2RGB(image['file'])
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
    for nMarker, marker in enumerate(markersNot):
        totalArea = 0
        quadrants = inA.imageQuadrantSplit(iO.notOperation(marker))
        for nQuadrant, q in enumerate(quadrants):
            sideOfImage = q[0].shape[0]
            _, contours, _ = cv2.findContours(
                q, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for nContour, contour in enumerate(contours):
                if inA.blobValidation(contour, nQuadrant,           sideOfImage):
                    totalArea += cv2.contourArea(contour)
                else:
                    print(f'Blob {nContour + 1} of quad {nQuadrant + 1} in marker {nMarker + 1} was not valid')
        print(totalArea / 490 * 100)
    mergedMarkers = inA.mergeQuadrants(markersNot)
    sP.showImage(mergedMarkers, figSize=(7, 7))
