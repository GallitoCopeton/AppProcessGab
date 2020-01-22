
import os
import sys

import cv2
from termcolor import colored
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Scripts para leer y procesar imagen
print(os.getcwd())
workingPath = os.getcwd()
scriptsPath = '../Golden Master (AS IS)'
os.chdir(scriptsPath)
import sorts as srt
import preProcessing as pP
import perspective as pPe
import indAnalysis as inA
os.chdir(workingPath)


# %%
imagesPath = './testImages/'
images = [image for image in os.listdir(imagesPath) if image.endswith('.jpg')]
mask = inA.readMask()
# %%

df = pd.DataFrame()
columns = ['ESAT6', 'CF', 'RV', 'Control', 'file name']
columns.reverse()
for imageNumber, imageName in enumerate(images):
    print(colored(imageName, 'green'))
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
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 1000, 100)
        k = 3
        attempts = 80
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

    results = {
        'file name': '',
        'ESAT6': 0,
        'CF': 0,
        'RV': 0,
        'Control': 0
    }
    fig = plt.figure()
    
    clusteredMarkers.reverse()
    for i, clusteredMarker in enumerate(clusteredMarkers):

        # Blobs in images
        ax1 = fig.add_subplot(1, 4, i+1)
        plt.subplots_adjust(right=1.2)
        
        blobsMarker = inA.blobDetect(clusteredMarker)
        ax1.imshow(inA.drawBlobs(clusteredMarker, blobsMarker))
        ax1.set_axis_off()
        clusteredMarker_area = 0
        clusteredMarker_blobs = inA.blobDetect(clusteredMarker)

        for clusteredMarker_blob in clusteredMarker_blobs:
            clusteredMarker_area += clusteredMarker_blob.size
        if i == 0:
            results['Control'] += clusteredMarker_area
            txt = f'Area of Control: {str(round(clusteredMarker_area, 2))}'
        elif i == 1:
            results['RV'] += clusteredMarker_area
            txt = f'Area of RV: {str(round(clusteredMarker_area, 2))}'
        elif i == 2:
            results['CF'] += clusteredMarker_area
            txt = f'Area of CF: {str(round(clusteredMarker_area, 2))}'
        elif i == 3:
            results['ESAT6'] += clusteredMarker_area
            txt = f'Area of ESAT6: {str(round(clusteredMarker_area, 2))}'
        results['file name'] = imageName[-10:]
        ax1.set_title(txt, fontsize=10)
        
    resultsDf = pd.DataFrame.from_records([results], columns=columns)
    df = df.append(resultsDf)
    plt.show()
# %%
csvFileName = 'Area Blobs TEST.csv'
csvFolderName = './resultados/'
if not os.path.isdir(csvFolderName):
    os.mkdir(csvFolderName)
df.to_csv(csvFolderName+csvFileName, mode='a')
