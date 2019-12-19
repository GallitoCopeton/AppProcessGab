import operator
import os
import sys

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pymongo

# Scripts para leer y procesar imagen
from ImageFunctions.ReadImages import readImage as rI
from ImageFunctions.ImageProcessing.colorTransformations import BGR2RGB
from ImageFunctions.ShowProcess import showProcesses as sP
from ImageFunctions.ImageProcessing import blobAnalysis as bA
from ImageFunctions.ImageProcessing import colorStats as cS
from ImageFunctions.ImageProcessing import imageOperations as iO
from QueryUtilities import qrQuery



# %%
imagesURI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
imagesDB = pymongo.MongoClient(imagesURI)['findValidation']

imageDetails = rI.customQueryDetails(
            imagesDB['imagestotals'], {'count':0}, order=1, limit=10)
# %%

df = pd.DataFrame()
columns = ['ESAT6', 'CF', 'RV', 'Control', 'QR', 'count', '_id']
for imageNumber, imageDict in enumerate(imageDetails):
    if 'qr' in imageDict.keys():    
        qrCode = imageDict['qr']
    else:
        qrCode = 'No QR'
    if 'count' in imageDict.keys(): 
        count = imageDict['count']
    else:
        count = 'No count'
    _id = imageDict['_id']
    print(qrCode)
    try:
        _, _, _, notMarkers = sP.showClusterProcess(imageDict['image'], 3, 25, (8, 9), True)
    except:
        print(f'Error ocurred with register _id: {_id}')
    imageName = f'{_id}-{qrCode}-{count}'
    results = {
        'Image name': '',
        'ESAT6': 0,
        'CF': 0,
        'RV': 0,
        'Control': 0
    }

    for i, notMarker in enumerate(notMarkers):

        notMarkerArea = bA.areaEstimation(bA.blobDetect(notMarker))
        notMarkerArea = cS.totalWhitePixels(iO.notOperation(notMarker))
        if i == 0:
            results['ESAT6'] += notMarkerArea
        elif i == 1:
            results['CF'] += notMarkerArea
        elif i == 2:
            results['RV'] += notMarkerArea
        elif i == 3:
            results['Control'] += notMarkerArea
        results['QR'] = qrCode
        results['count'] = count
        results['_id'] = _id
    resultsDf = pd.DataFrame.from_records([results], columns=columns)
    df = df.append(resultsDf)
df.set_index('QR', inplace=True)
#%%
dfESAT = df[(df['ESAT6'] > 25) & (df['Control'] < 25)]