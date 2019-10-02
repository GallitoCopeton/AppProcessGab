# %%


import sys
import os

import cv2
import numpy as np
import pymongo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, '../Golden Master (AS IS)')
try:
    os.chdir(os.path.join(os.getcwd(), 'Golden Master (AS IS)/'))
    print(os.getcwd())
except:
    print(os.getcwd())

import indAnalysis as inA
import preProcessing as pP
import readImage
print('Scripts imported')
# %% Read all the images from the database prodLaboratorio in collection markerTotals that have diagnosis excluding P24
MONGO_URL = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
client = pymongo.MongoClient(MONGO_URL)
db = client.prodLaboratorio
markerTotals = db.markerTotals
markers = markerTotals.find({'diagnostic': {'$ne': None},
                             'marker': {'$ne': 'P24'}}, {'count': 0, 'QR': 0, '_id': 0})
mask = inA.readMask(url='../Imagenes/mask_inv.png')
dfDict = {}
dfForTraining = pd.DataFrame()
# %%
for i, marker in enumerate(markers):
    # Base 64 => 3 channel file
    imageBase64 = marker['image']
    originalImage = inA.resizeFixed(readImage.readb64(imageBase64))

    # Image preprocessing => START

    # Reshape image to have the R G B features (3) each as a column (-1)
    Z = originalImage.reshape((-1, 3))
    # Convert to np.float32
    Z = np.float32(Z)
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, centers = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8 and make original image
    centers = np.uint8(centers)
    kMeansReconstructed = centers[label.flatten()]
    kMeansReconstructedReshaped = kMeansReconstructed.reshape(
        (originalImage.shape))
    kMeansReconstructedReshaped = pP.median(
        kMeansReconstructedReshaped, 1)
    # Grayscale image
    imgGray = cv2.cvtColor(
        kMeansReconstructedReshaped, cv2.COLOR_BGR2GRAY)
    # Thresholded image
    _, imgBina = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
    # And operation with mask
    imgBinaMask = inA.andOperation(imgBina, mask)
    # Opening of the final image (Erotion followed by dilation)
    kernel = np.ones((1, 1), np.uint8)
    imgBinaMaskOpen = cv2.morphologyEx(
        imgBinaMask, cv2.MORPH_OPEN, kernel)

    # Image preprocessing => FINISH

    # Blob analysis => START
    quadrantBlobResults = inA.quadrantBlobAnalisys(
        imgBinaMaskOpen, kMeansReconstructedReshaped)
    dfDict.update(quadrantBlobResults)
    try:
        dfDict.update({
            'diagnostic': marker['diagnostic'].upper()
        })
    except:
        continue
    dictValues = list(dfDict.values())
    diagnostic = dictValues.pop()
    dictValues = np.array(dictValues, dtype='int32')
    if dictValues.sum() == 0 and diagnostic == 'P':
        continue
    dfForTraining = dfForTraining.append(dfDict, ignore_index=True)
    # Blob analysis => FINISH

    # Image shows (Only one per execution)

    # Show original image
    #plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))

    # Show reconstructed image
    #plt.imshow(cv2.cvtColor(kMeansReconstructedReshaped, cv2.COLOR_BGR2RGB))

    # Show grayscale image
    #plt.imshow(imgGray)

    # Show thresholded image
    #plt.imshow(imgBina, 'gray')

    # Show masked image
    #plt.imshow(imgBinaMask, 'gray')

    # Show final blobs
    #plt.imshow(imgBinaMaskOpen, 'gray')
    d = dfDict['diagnostic']
    plt.show()
    print(f'Diagnostic: {d}')
# %%
    dfForTrainingCopy = dfForTraining
    print(dfForTraining)
# %% Bitch dataframe is not what it is supposed to be, filter it
for i in range(0, 4):
    dfForTraining.loc[dfForTraining[f'distanceQ{i+1}']
                      < 4, f'areaQ{i+1}'] = 0
# %%
dfForTraining.to_csv('../Dataframes/MoreFeatures.csv')
# %%
sns.jointplot(y=dfForTraining['horizontalDiagQ1'],x=dfForTraining['verticalDiagQ1'],data=dfForTraining)
# %%
del dfForTraining['totalArea']
