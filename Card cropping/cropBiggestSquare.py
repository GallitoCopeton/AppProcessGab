# %%
import os

import cv2

import qrQuery
from AppProcess.CroppingProcess import croppingProcess as cP
from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
# %%
URI = 'mongodb+srv://findOnlyReadUser:RojutuNHqy@clusterfinddemo-lwvvo.mongodb.net/datamap?retryWrites=true'
collectionNameImages = 'imagestotals'
dbNameImages = 'datamap'
collectionImages = qrQuery.getCollection(
    URI, dbNameImages, collectionNameImages)
with open('alreadyQueriedQRs.txt', 'r') as file:
    alreadyQueriedQRs = [line for line in file]
alreadyQueriedQRs = [qrQuery.fixQr(QR) for QR in alreadyQueriedQRs]
query = {'fileName': {'$nin': alreadyQueriedQRs}}
# %% Cluster preprocessing approach
criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100000, 100000)
notMeetCondition = 0
cloudImages = rI.customQuery(collectionImages, query, limit=1, order=-1)
localImages = [rI.readLocal(image) for image in os.listdir(
    './') if image.endswith('.jpg') or image == '0.png']
localImages = []
images = cloudImages + localImages
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
# %%
for image in images:
    # Original image loading and resizing
    try:
        with open('alreadyQueriedQRs.txt', 'a') as file:
            print('Will append: {}'.format(image['fileName']))
            file.writelines(image['fileName']+'\n')
        image = image['file']
    except:
        pass
    # FINALLY THE TEST AREA
    try:
        testArea = cP.getTestArea(image)
    except Exception as e:
        print(e)
        continue
    markers = cP.getMarkersV2(testArea)
    if markers is not None:
        [sP.showImage(marker, figSize=(2, 2)) for marker in markers]
