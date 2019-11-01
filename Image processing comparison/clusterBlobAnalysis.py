import operator
import os
import sys

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Scripts para leer y procesar imagen
sys.path.insert(0, '../Golden Master (AS IS)')
import indAnalysis as inA
import perspective as pPe
import preProcessing as pP
import readImage as rI
import sorts as srt




# %%
#imagesPath = './Positives/'
imagesPath = './Negatives/'
images = os.listdir(imagesPath)
mask = inA.readMask()
# %%

df = pd.DataFrame()
columns = ['ESAT6', 'CF', 'RV', 'Control', 'file name']
for imageNumber, imageName in enumerate(images):
    print(imageName)
    # Normal process for both (individual extraction)
    imagePath = os.path.join(imagesPath, imageName)
    image = cv2.imread(imagePath)
    image = pP.resizeImg(image, 728)
    imgBinary = pP.contourBinarization(
        image, 3, 7, 85, 2, inverse=True, mean=False)
    externalSquare = pP.findTreeContours(imgBinary)
    ext_index = 0
    externalOrdSquare = srt.sortPointsContours(externalSquare)
    perspectiveBinary = pPe.perspectiveTransform(
        imgBinary, externalOrdSquare[ext_index], -5, True)
    perspectiveBGR = pPe.perspectiveTransform(
        image, externalOrdSquare[ext_index], -5)
    external = pP.findExternalContours(perspectiveBinary, area=115000)
    testSquareBGR = pPe.getTestSquare(perspectiveBGR, external)
    testSquareBinary = pPe.getTestSquare(perspectiveBinary, external, True)
    testSquareBGRCopy = testSquareBGR.copy()
    height, width = testSquareBGR.shape[:2]
    areaInd = (height*width/8) - 10
    # Binarization is required again because of the loss of details on first one
    contoursInd = pP.findTreeContours(pP.contourBinarization(
        testSquareBGR, 3, 7, 85, 2, mean=False), 115000)
    # For tests with 4 or 6 squares
    if(len(contoursInd) == 5 or len(contoursInd) == 7):
        contoursInd = contoursInd[1:]
    srt.sortTests(contoursInd)
    listMarkers = []
    if(len(contoursInd) == 4 or len(contoursInd) == 6):
        for i, c in enumerate(contoursInd):
            marker = pPe.getIndTest(testSquareBGR, c)
            listMarkers.append(marker)
    listMarkers = inA.resizeAll(listMarkers)
    clusteredMarkers = []
    for j, marker in enumerate(listMarkers):
        # Clusters start
        print(f'Marker no. {j+1}')
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10000, 1000)
        k = 5
        attempts = 100
        reconMarker = pP.clusterReconstruction(marker, criteria, k, attempts)
        # Grayscale image
        grayMarker = cv2.cvtColor(
            reconMarker, cv2.COLOR_BGR2GRAY)
        # Thresholded image
        _, binMarker = cv2.threshold(grayMarker, 150, 255, cv2.THRESH_BINARY)
        # And operation with mask
        maskMarker = inA.andOperation(binMarker, mask)
        # K Means processing (app)
        kernelOpen = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernelDila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        transMarker = cv2.morphologyEx(
            maskMarker, cv2.MORPH_OPEN, kernelOpen)
        transMarker = cv2.morphologyEx(
            transMarker, cv2.MORPH_DILATE, kernelDila)
        notMarker = cv2.bitwise_not(transMarker)
        clusteredMarkers.append(notMarker)

        # Blobs in images
        blobsMarker = inA.blobDetect(notMarker)
        plt.subplot(121)
        plt.imshow(inA.drawBlobs(notMarker, blobsMarker), 'gray')
        plt.subplot(122)
        plt.imshow(reconMarker)
        plt.title('Original Image')
        plt.show()

    results = {
        'file name': '',
        'ESAT6': 0,
        'CF': 0,
        'RV': 0,
        'Control': 0
    }

    for i, clusteredMarker in enumerate(clusteredMarkers):

        clusteredMarker_area = 0
        clusteredMarker_blobs = inA.blobDetect(clusteredMarker)

        for clusteredMarker_blob in clusteredMarker_blobs:
            clusteredMarker_area += clusteredMarker_blob.size
        if i == 0:
            results['ESAT6'] += clusteredMarker_area
        elif i == 1:
            results['CF'] += clusteredMarker_area
        elif i == 2:
            results['RV'] += clusteredMarker_area
        elif i == 3:
            results['Control'] += clusteredMarker_area
        results['file name'] = imageName[-10:]
    resultsDf = pd.DataFrame.from_records([results], columns=columns)
    df = df.append(resultsDf)
