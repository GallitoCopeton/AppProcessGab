# %%
import os
import datetime
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ReadImages import readImage as rI
from AppProcess.MarkerProcess import markerProcess as mP
from ImageProcessing import imageOperations as iO
from ImageProcessing import colorStats as cS
from ImageProcessing import blobAnalysis as bA
from ImageProcessing import indAnalysis as inA
import qrQuery
# %% Read all the images from the database prodLaboratorio in collection markerTotals that have diagnosis excluding P24
URI = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
dbName = 'prodLaboratorio'
collectionName = 'markerTotals'
collection = qrQuery.getCollection(URI, dbName, collectionName)

query = {'diagnostic': {'$ne': None},
         'marker': {'$ne': 'P24'}}
# %%
markers = collection.find(query).limit(0)
markerInfo = ([(iO.resizeFixed(rI.readb64(marker['image'])),
                marker['diagnostic']) for marker in markers])
markerImages = [info[0] for info in markerInfo]
markerDiagnostic = [info[1] for info in markerInfo]
NOTMarkers = mP.clusteringProcess(markerImages, 3, 2)
# %%
infoDict = {
    'whitePixels': [],
    'blobDiameter': [],
    'nBlobs': [],
    'diagnostic': []
}
i = 0
infoDataframes = []
for (marker, diagnostic, oMarker) in zip(NOTMarkers, markerDiagnostic, markerImages):
    print(i)
    i += 1
    whitePixels = cS.totalWhitePixels(iO.notOperation(marker))
    blobs = bA.blobDetect(marker)
    blobDiameter = bA.areaEstimation(blobs)
    nBlobs = len(blobs)
    info = inA.deepQuadrantAnalysis(iO.notOperation(marker))
    try:
        infoDict['diagnostic'].append(1 if diagnostic.upper() == 'P' else 0)
        info['diagnostic'] = 1 if diagnostic.upper() == 'P' else 0
    except AttributeError as e:
        print(e)
        continue
    infoDict['whitePixels'].append(whitePixels)
    infoDict['blobDiameter'].append(blobDiameter)
    infoDict['nBlobs'].append(nBlobs)
    tempDf = pd.DataFrame.from_dict(info, orient='index').T
    infoDataframes.append(tempDf)
infoDataframe = pd.DataFrame().from_dict(infoDict)
infoDataframeExtended = pd.concat(infoDataframes, axis=0)
# %%
todaysDate = datetime.datetime.now()
dateString = re.sub(r':', '_', todaysDate.ctime())[4:]
currentFolder = os.path.dirname(os.path.realpath(__file__))
dataframesFolder = 'csvs'
folderPath = f'{currentFolder}/{dataframesFolder}'
qrQuery.makeFolders(folderPath)
infoCsvName = '/'.join([folderPath, f'Dataframe de {dateString}.csv'])
infoDataframeExtended.to_csv(infoCsvName, index=False)
