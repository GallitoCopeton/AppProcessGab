# %%
import os
import sys

import cv2
import pymongo
from matplotlib import pyplot as plt

try:
    os.chdir(os.path.join(os.getcwd(), 'Helper notebooks'))
    print(os.getcwd())
except:
    print(os.getcwd())
# Scripts para leer y procesar imagen
sys.path.insert(0, '../Golden Master (AS IS)')
import indAnalysis as inA
import preProcessing as pP
import readImage

# %% [markdown]
# ## Golden Master Image Algorithm (AS IS)
# %% [markdown]
# ### Read Image

# %% Read all the images from the database prodLaboratorio in collection markerTotals that have diagnosis
MONGO_URL = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
client = pymongo.MongoClient(MONGO_URL)
db = client.prodLaboratorio
markerTotals = db.markerTotals
markers = markerTotals.find({'diagnostic': {'$ne': None},
                             'marker': {'$ne': 'P24'}}, {'count': 0, 'QR': 0, '_id': 0})
count = 1
mask = inA.readMask()
results = {
    'coincidences': 0,
    'errors': 0,
}

for imageNumber, marker in enumerate(markers):
    imageBase64 = marker['image']
    originalImage = inA.resizeFixed(readImage.readb64(imageBase64))
    if count > 0:
        if(isinstance(originalImage, str)):
            print(originalImage)
            continue
        else:
            # Image preprocessing => START
            imgBina = pP.contourBinarizationOtsu(
                originalImage, 3, 3, 45, 4, Gs=0, inverse=False, mean=True)
            imgBinMask = inA.andOperation(imgBina, mask)
            imgBinMaskMorph = inA.erosionDilation(imgBinMask, 3)
            imgBinMaskMorphInv = cv2.bitwise_not(imgBinMaskMorph)
            # Image preprocessing => FINISH

            # Blob analysis => START
            imgBinaMaskInvMorphCopy = imgBinMaskMorphInv.copy()
            blobs = inA.blobDetect(imgBinMaskMorphInv)
            area = round(inA.areaEstimation(blobs), 2)
            diagnosis = inA.areaAnalysis2(
                imgBina, blobs, area, marker['marker'])
            # Blob analysis => FINISH

            # Diagnosis comparison => START
            blobsStats = {
                'diagnosis': diagnosis
            }
            dbStats = {
                'diagnosis': marker['diagnostic']
            }
            if (blobsStats['diagnosis'] == dbStats['diagnosis']):
                results['coincidences'] += 1
            elif (blobsStats['diagnosis'] != dbStats['diagnosis']):
                results['errors'] += 1
            # Diagnosis comparison => FINISH
            
            # Image shows (Only one per execution)

            # Show original image
            #plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))

            # Show thresholded image
            #plt.imshow(imgBina, 'gray')

            # Show masked image
            #plt.imshow(imgBinaMask, 'gray')

            # Show final blobs
            #plt.imshow(imgBinMaskMorphInv, 'gray')

            # Draw the detected blobs
            #plt.imshow(inA.drawBlobs(imgBinaMaskInvMorphCopy, blobs), 'gray')
            
            #plt.show()
    else:
        break
    count += 1
# %%
print(results)
accuracy = results['coincidences']/(results['coincidences'] + results['errors'])
print(f'Accuracy: {accuracy}')
