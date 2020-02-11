# %%
import os
import json
from datetime import datetime
from datetime import timedelta

from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.ReadImage import readImage as rI
from IF2.Crop import croppingProcess as cP
from IF2.Marker import markerProcess as mP
from IF2.Shows.showProcesses import showImage as show
from IF2.Shows import showProcesses as sP
import qrQuery

#%%
with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
xploraStaging = connections['stagingXplora']
xploraStagingImagesCollection = qrQuery.getCollection(
    xploraStaging['URI'], xploraStaging['databaseName'], xploraStaging['collections']['imagesCollectionName'])
# %%
todaysDate = datetime.now()
startDay = -1
finishDay = 3
startDate = todaysDate - timedelta(days=startDay)
finishDate = startDate - timedelta(days=finishDay-startDay)
dateQuery = qrQuery.getDateQuery(startDate, finishDate)
imageDocuments = xploraStagingImagesCollection.find(dateQuery)
imagesInfo = [[(iO.resizeImg(rI.readb64(image['file']))),
                {'qr': image['fileName'],
                 'count': image['count'],
                 '_id': image['_id']}
                ] for image in imageDocuments]
images = [info[0] for info in imagesInfo]
imagesInfo = [info[1] for info in imagesInfo]
#%%

for imageNumber, (image, imageInfo) in enumerate(zip(images, imagesInfo)):
    qr = imageInfo['qr']
    count = imageInfo['count']
    print(f'register QR: {qr} COUNT: {count}')
    try:
        _, _, _, notMarkers = sP.showClusterProcess(image, 3, 6, figSizeTupple=(9,9))
    except Exception as e:
        print(e)
        print(f'Error ocurred with register QR: {qr} COUNT: {count}')