# %%

import sorts as srt
import readImage as rI
import preProcessing as pP
import perspective as pPe
import indAnalysis as inA
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
#imagesPath = './Positives/'
imagesPath = './Negatives/'
images = os.listdir(imagesPath)
mask = inA.readMask()

# %%

df = pd.DataFrame()
for imageNumber, imageName in enumerate(images):
    print(imageName)
    # Normal process for both (individual extraction)
    imagePath = os.path.join(imagesPath, imageName)
    image = cv2.imread(imagePath)
    image = pP.resizeImg(image, 728)
    # Plot original
    # plt.subplot(111),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # plt.show()
    imgBinary = pP.contourBinarization(
        image, 3, 7, 85, 2, inverse=True, mean=False)
    # Plot first binary
    # plt.subplot(111),plt.imshow(imgBinary, 'gray')
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
    external = pP.findExternalContours(perspectiveBinary, area=115000)
    testSquareBGR = pPe.getTestSquare(perspectiveBGR, external)
    testSquareBGR = pP.equalizeHistogram(testSquareBGR, claheEq=True)
    testSquareBinary = pPe.getTestSquare(perspectiveBinary, external, True)
    testSquareBGRCopy = testSquareBGR.copy()
    height, width = testSquareBGR.shape[:2]
    areaInd = (height*width/8) - 10
    # Binarization is required again because of the loss of details on first one
    contoursInd = pP.findTreeContours(pP.contourBinarization(
        testSquareBGR, 3, 7, 85, 2, mean=False), 115000)
    if(len(contoursInd) == 5 or len(contoursInd) == 7):
        contoursInd = contoursInd[1:]
    for c in contoursInd:
        cv2.drawContours(testSquareBGRCopy, [c], -1, (0, 255, 0), 3)
    # Plot first perspective
    # plt.subplot(111), plt.imshow(testSquareBGRCopy)
    # plt.show()
    srt.sortTests(contoursInd)
    listTests = []
    if(len(contoursInd) == 4 or len(contoursInd) == 6):
        for i, c in enumerate(contoursInd):
            test = pPe.getIndTest(testSquareBGR, c)
            listTests.append(test)
    listTests = inA.resizeAll(listTests)
    kMeansOriginal = []
    kMeansApp = []
    areasOriginal = []
    areasApp = []
    for j, test in enumerate(listTests):
        # Kmeans start
        print(f'Marker no. {j+1}')
        Z = test.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        K = 3
        ret, label, centers = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        kMeansReconstructed = centers[label.flatten()]
        kMeansReconstructedReshaped = kMeansReconstructed.reshape(
            (test.shape))
        # Grayscale image
        imgGray = cv2.cvtColor(
            kMeansReconstructedReshaped, cv2.COLOR_BGR2GRAY)
        # Thresholded image
        _, imgBina = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
        # And operation with mask
        imgBinaMask = inA.andOperation(imgBina, mask)
        # K Means processing (original)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        imgBinaMaskDil = cv2.morphologyEx(
            imgBinaMask, cv2.MORPH_DILATE, kernel1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        imgBinaMaskDilEro = cv2.morphologyEx(
            imgBinaMaskDil, cv2.MORPH_OPEN, kernel2)
        imgBinaMaskDil = cv2.bitwise_not(imgBinaMaskDilEro)
        kMeansOriginal.append(imgBinaMaskDil)
        # K Means processing (actual app)
        imgBina = pP.contourBinarizationOtsu(
            kMeansReconstructedReshaped, 3, 3, 45, 3, Gs=0, inverse=False, mean=True)
        imgBinaMask = inA.andOperation(imgBina, mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erodeKMeans = cv2.erode(imgBinaMask, kernel)
        dilateKMeans = cv2.dilate(erodeKMeans, kernel, iterations=2)
        imgBinaMaskEroDilInv = cv2.bitwise_not(dilateKMeans)
        kMeansApp.append(imgBinaMaskEroDilInv)
        # Kmeans finish
        # Areas start
        testBin = pP.contourBinarizationOtsu(
            test, 3, 3, 45, 3, Gs=0, inverse=False, mean=True)
        testBinMask = inA.andOperation(testBin, mask)
        # Areas processing (python original)
        testEroDil = inA.erosionDilation(testBinMask, 3)
        testEroDilInvOrig = cv2.bitwise_not(testEroDil)
        areasOriginal.append(testEroDilInvOrig)
        # Areas processing (actual app)

        erode = cv2.erode(testBinMask, kernel)
        dilate = cv2.dilate(erode, kernel, iterations=2)
        testEroDilInvApp = cv2.bitwise_not(dilate)
        areasApp.append(testEroDilInvApp)

        # Blobs in images
        kOrig_blobs = inA.blobDetect(imgBinaMaskDil)
        kApp_blobs = inA.blobDetect(imgBinaMaskEroDilInv)
        areaOrig_blobs = inA.blobDetect(testEroDilInvOrig)
        areaApp_blobs = inA.blobDetect(testEroDilInvApp)
        # Areas finish
        plt.subplot(121)
        plt.imshow(inA.drawBlobs(imgBinaMaskDil, kOrig_blobs), 'gray')
#        plt.title('kMean Original')
#        plt.subplot(152)
#        plt.imshow(inA.drawBlobs(imgBinaMaskEroDilInv, kApp_blobs), 'gray')
#        plt.title('kMean App')
#        plt.subplot(153)
#        plt.imshow(inA.drawBlobs(testEroDilInvOrig, areaOrig_blobs), 'gray')
#        plt.title('Areas Original')
#        plt.subplot(154)
#        plt.imshow(inA.drawBlobs(testEroDilInvApp, areaApp_blobs), 'gray')
#        plt.title('Areas App')
        plt.subplot(122)
        plt.imshow(kMeansReconstructedReshaped)
        plt.title('Original Image')
        plt.show()
# %%
    results = {
        'fileName': '',
        'kMeansOriginal': {
            'ESAT6': 0,
            'CF': 0,
            'RV': 0,
            'Control': 0
        },
        'kMeansApp': {
            'ESAT6': 0,
            'CF': 0,
            'RV': 0,
            'Control': 0
        },
        'areasOriginal': {
            'ESAT6': 0,
            'CF': 0,
            'RV': 0,
            'Control': 0
        },
        'areasApp': {
            'ESAT6': 0,
            'CF': 0,
            'RV': 0,
            'Control': 0
        }
    }
    iList = np.arange(0, 4)
    for kOrig, kApp, areaOrig, areaApp, i in zip(kMeansOriginal, kMeansApp, areasOriginal, areasApp, iList):

        kOrig_area = 0
        kApp_area = 0
        areaOrig_area = 0
        areaApp_area = 0

        kOrig_blobs = inA.blobDetect(kOrig)
        kApp_blobs = inA.blobDetect(kApp)
        areaOrig_blobs = inA.blobDetect(areaOrig)
        areaApp_blobs = inA.blobDetect(areaApp)

        for kOrig_blob in kOrig_blobs:
            kOrig_area += kOrig_blob.size
        if i == 0:
            results['kMeansOriginal']['ESAT6'] += kOrig_area
        elif i == 1:
            results['kMeansOriginal']['CF'] += kOrig_area
        elif i == 2:
            results['kMeansOriginal']['RV'] += kOrig_area
        elif i == 3:
            results['kMeansOriginal']['Control'] += kOrig_area

        for kApp_blob in (kApp_blobs):
            kApp_area += kApp_blob.size
        if i == 0:
            results['kMeansApp']['ESAT6'] += kApp_area
        elif i == 1:
            results['kMeansApp']['CF'] += kApp_area
        elif i == 2:
            results['kMeansApp']['RV'] += kApp_area
        elif i == 3:
            results['kMeansApp']['Control'] += kApp_area

        for areaOrig_blob in areaOrig_blobs:
            areaOrig_area += areaOrig_blob.size
        if i == 0:
            results['areasOriginal']['ESAT6'] += areaOrig_area
        elif i == 1:
            results['areasOriginal']['CF'] += areaOrig_area
        elif i == 2:
            results['areasOriginal']['RV'] += areaOrig_area
        elif i == 3:
            results['areasOriginal']['Control'] += areaOrig_area

        for areaApp_blob in areaApp_blobs:
            areaApp_area += areaApp_blob.size
        if i == 0:
            results['areasApp']['ESAT6'] += areaApp_area
        elif i == 1:
            results['areasApp']['CF'] += areaApp_area
        elif i == 2:
            results['areasApp']['RV'] += areaApp_area
        elif i == 3:
            results['areasApp']['Control'] += areaApp_area

        results['fileName'] = imageName[-10:]
    resultsDf = pd.DataFrame.from_dict(results)
    df = df.append(resultsDf)
# %%
positivesFileName = 'positives.xlsx'
negativesFileName = 'negatives.xlsx'

df.to_excel(negativesFileName)
