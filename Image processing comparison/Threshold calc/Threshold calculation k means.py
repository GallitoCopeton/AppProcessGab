# %%
import indAnalysis as inA
import perspective as pPe
import preProcessing as pP
import readImage as rI
import sorts as srt
import operator
import os
import sys

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    os.chdir(os.path.join(os.getcwd(), 'Helper notebooks'))
    print(os.getcwd())
except:
    print(os.getcwd())
# Scripts para leer y procesar imagen
sys.path.insert(0, '../Golden Master (AS IS)')
# %%
imagesPath = './img14/'
images = os.listdir(imagesPath)
mask = inA.readMask()

# %%

df = pd.DataFrame()
for imageNumber, imageName in enumerate(images):
    imagePath = os.path.join(imagesPath, imageName)
    image = cv2.imread(imagePath)
    image = pP.resizeImg(image, 728)
    # Plot original
    # plt.subplot(111),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # plt.show()
    imgBinary = pP.contourBinarization(
        image, 3, 7, 85, 2, inverse=True, mean=False)
    # Plot first binary
    #plt.subplot(111),plt.imshow(imgBinary, 'gray')
    # plt.show()
    externalSquare = pP.findTreeContours(imgBinary)
    ext_index = 0
    externalOrdSquare = srt.sortPointsContours(externalSquare)
    perspectiveBinary = pPe.perspectiveTransform(
        imgBinary, externalOrdSquare[ext_index], -5, True)
    perspectiveBGR = pPe.perspectiveTransform(
        image, externalOrdSquare[ext_index], -5)
    # Plot first perspective
    # plt.subplot(111),plt.imshow(perspectiveBGR)
    # plt.show()
    external = pP.findExternalContours(
        perspectiveBinary, area=115000)
    testSquareBGR = pPe.getTestSquare(perspectiveBGR, external)
    testSquareBGR = pP.equalizeHistogram(testSquareBGR)
    testSquareBinary = pPe.getTestSquare(perspectiveBinary, external, True)
    testSquareBGRCopy = testSquareBGR.copy()
    height, width = testSquareBGR.shape[:2]
    areaInd = (height*width/8) - 10
    # Binarization is required again because of the loss of details on first one
    contoursInd = pP.findTreeContours(pP.contourBinarization(
        testSquareBGR, 3, 7, 85, 2, mean=False), 115000)
    if(len(contoursInd) == 5):
        contoursInd = contoursInd[1:]
    for c in contoursInd:
        cv2.drawContours(testSquareBGRCopy, [c], -1, (0, 255, 0), 3)
    # Plot first perspective
    #plt.subplot(111), plt.imshow(testSquareBGRCopy)
    #plt.show()
    srt.sortTests(contoursInd)
    listTests = []
    if(len(contoursInd) == 4):
        for i, c in enumerate(contoursInd):
            test = pPe.getIndTest(testSquareBGR, c)
            listTests.append(test)
    listTests = inA.resizeAll(listTests)
    maskedImages = []
    # K Means processing
    for j, test in enumerate(listTests):
        print(f'Marker no. {j+1}')
        Z = test.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret, label, centers = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        kMeansReconstructed = centers[label.flatten()]
        kMeansReconstructedReshaped = kMeansReconstructed.reshape(
            (test.shape))
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
        imgBinaMaskOpen = cv2.bitwise_not(imgBinaMaskOpen)
        maskedImages.append(imgBinaMaskOpen)
        plt.subplot(131)
        plt.imshow(kMeansReconstructedReshaped)
        plt.subplot(132)
        plt.imshow(test)
        plt.subplot(133)
        plt.imshow(imgBinaMaskOpen, 'gray')
        plt.show()
    markerArea = {
            'ESAT6': 0,
            'CF': 0,
            'RV': 0,
            'Control':0
        }
    for i, image in enumerate(maskedImages):
        area = 0
        blobs = inA.blobDetect(image)
        for blob in blobs:
            area += blob.size
        if i == 0:
            markerArea['ESAT6'] += area
        elif i == 1:
            markerArea['CF'] += area
        elif i == 2:
            markerArea['RV'] += area
        elif i == 3:
            markerArea['Control'] += area
    df = df.append(markerArea, ignore_index=True)
        
# %%
ESATMean = df['ESAT6'].mean()
CFMean = df['CF'].mean()
RVMean = df['RV'].mean()
ControlMean = df['Control'].mean()
#%%
fileName = 'WhitePixelsKMeans.xlsx'
df.to_excel(fileName)