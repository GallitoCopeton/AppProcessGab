import math
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Scripts para leer y procesar imagen
print(os.getcwd())
workingPath = os.getcwd()
scriptsPath = '../Golden Master (AS IS)'
os.chdir(scriptsPath)
import indAnalysis as inA
import perspective as pPe
import preProcessing as pP
import sorts as srt
os.chdir(workingPath)

def showImages(imagesList):
    for image in imagesList:
        plt.imshow(image)
        plt.show()

# %%
imagesPath = './testImages/'
images = os.listdir(imagesPath)
mask = inA.readMask()
# %%
clusterImagesFolder = './clusterImagesPython/'
if not os.path.isdir(clusterImagesFolder):
    os.mkdir(clusterImagesFolder)
for imageNumber, imageName in enumerate(images):
    processImages = []
    print(imageName)
    # Normal process for both (individual extraction)
    imagePath = os.path.join(imagesPath, imageName)
    image = cv2.imread(imagePath)
    image = pP.resizeImg(image, 728)
    imgBinary = pP.contourBinarization(
        image, 3, 7, 85, 2, inverse=True, mean=False)
    processImages.append(imgBinary)
    externalSquare = pP.findTreeContours(imgBinary)
    ext_index = 0
    externalOrdSquare = srt.sortPointsContours(externalSquare)
    perspectiveBinary = pPe.perspectiveTransform(
        imgBinary, externalOrdSquare[ext_index], -5, True)
    perspectiveBGR = pPe.perspectiveTransform(
        image, externalOrdSquare[ext_index], -5)
    processImages.append(perspectiveBGR)
    external = pP.findExternalContours(perspectiveBinary, area=115000)
    testSquareBGR = pPe.getTestSquare(perspectiveBGR, external)
    testSquareBinary = pPe.getTestSquare(perspectiveBinary, external, True)
    testSquareBGRCopy = testSquareBGR.copy()
    processImages.append(testSquareBGRCopy)
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
    listMarkers.reverse()
    listMarkers = inA.resizeAll(listMarkers)
    for j, marker in enumerate(listMarkers):
        # Clusters start
        print(f'Marker no. {j+1}')
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 1000, 100)
        k = 3
        attempts = 80
        reconMarker = pP.clusterReconstruction(
            marker, criteria, k, attempts)
        markerName = f'PYTHON_{imageName}_marker_{j+1}.png'
        plt.imsave(clusterImagesFolder + markerName, reconMarker)