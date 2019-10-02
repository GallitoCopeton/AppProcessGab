# %%
import json
import os
import sys

import cv2
import numpy as np
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
                             'marker': {'$ne': 'P24'}})
count = 1
mask = inA.readMask()
results = {
    'coincidences': 0,
    'errors': 0
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
            # Reshape image to have the R G B features (3) each as a column (-1)
            Z = originalImage.reshape((-1, 3))
            # Convert to np.float32
            Z = np.float32(Z)
            # Define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
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
            _, imgBina = cv2.threshold(imgGray, 160, 255, cv2.THRESH_BINARY)
            # And operation with mask
            imgBinaMask = inA.andOperation(imgBina, mask)
            # Opening of the final image (Erotion followed by dilation)
            kernel = np.ones((1, 1), np.uint8)
            imgBinaMaskOpen = cv2.morphologyEx(
                imgBinaMask, cv2.MORPH_OPEN, kernel)
            # Image preprocessing => FINISH

            # Blob analysis => START
            blobs = inA.blobDetect(imgBinaMaskOpen)
            area = round(inA.areaEstimation(blobs), 2)
            diagnosis = inA.areaAnalysis2(
                imgBina, blobs, area, marker['marker'])
            # Blob analysis => FINISH

            # Diagnosis comparison => START
            kMeansStats = {
                'diagnosis': diagnosis
            }
            dbStats = {
                'diagnosis': marker['diagnostic']
            }
            if (kMeansStats['diagnosis'] == dbStats['diagnosis']):
                results['coincidences'] += 1
            elif (kMeansStats['diagnosis'] != dbStats['diagnosis']):
                results['errors'] += 1
            # Diagnosis comparison => FINISH

            # Image shows (Only one per execution)
        
            # Show original image
            #plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
        
            # Show reconstructed image
            #plt.imshow(cv2.cvtColor(kMeansReconstructedReshaped, cv2.COLOR_BGR2RGB))
        
            # Show grayscale image
            # plt.imshow(imgGray)
        
            # Show thresholded image
            #plt.imshow(imgBina, 'gray')
        
            # Show masked image
            #plt.imshow(imgBinaMask, 'gray')
        
            # Show final blobs
            #plt.imshow(imgBinaMaskOpen, 'gray')
            
            # plt.show()
    else:
        break
# %%
print(results)
accuracy = results['coincidences']/(results['coincidences'] + results['errors'])
print(f'Accuracy: {accuracy}')

outputFile = 'Results_Object_Areas2.json'
file = open(outputFile, 'w')
file.write(json.dumps(results, indent=2))
file.close()
