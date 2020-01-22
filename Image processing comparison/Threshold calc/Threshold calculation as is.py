#%% 
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
import sorts as srt
import readImage as rI
import preProcessing as pP
import perspective as pPe
import indAnalysis as inA
#%%
imagesPath = './Photos/'
images = os.listdir(imagesPath)[13:]
mask = inA.readMask()

#%%
df = pd.DataFrame()
for imageNumber, imageName in enumerate(images):
    imagePath = os.path.join(imagesPath, imageName)
    image = cv2.imread(imagePath)
    image = pP.resizeImg(image, 728)
    #Plot original
    #plt.subplot(111),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    #plt.show()
    imgBinary = pP.contourBinarization(image, 3, 7, 85, 2, inverse = True, mean = False)
    #Plot first binary
    #plt.subplot(111),plt.imshow(imgBinary, 'gray')
    #plt.show()
    externalSquare = pP.findTreeContours(imgBinary)
    ext_index = 0
    externalOrdSquare = srt.sortPointsContours(externalSquare)
    perspectiveBinary = pPe.perspectiveTransform(imgBinary, externalOrdSquare[ext_index], -5, True)
    perspectiveBGR = pPe.perspectiveTransform(image, externalOrdSquare[ext_index], -5)
    #Plot first perspective
    #plt.subplot(111),plt.imshow(perspectiveBGR)
    #plt.show()
    external = pP.findExternalContours(
        perspectiveBinary, area=115000)
    testSquareBGR = pPe.getTestSquare(perspectiveBGR, external)
    testSquareBGR = pP.equalizeHistogram(testSquareBGR)
    testSquareBinary = pPe.getTestSquare(perspectiveBinary, external, True) 
    testSquareBGRCopy = testSquareBGR.copy()
    height, width = testSquareBGR.shape[:2]
    areaInd = (height*width/8) - 10
    #Binarization is required again because of the loss of details on first one
    contoursInd = pP.findTreeContours(pP.contourBinarization(testSquareBGR, 3, 7, 85, 2, mean = False),115000)
    if(len(contoursInd) == 5 ): 
        contoursInd = contoursInd[1:]
    for c in contoursInd:
        cv2.drawContours(testSquareBGRCopy, [c], -1, (0, 255, 0), 3)
    #Plot first perspective
    plt.subplot(111),plt.imshow(testSquareBGRCopy)
    plt.show()
    srt.sortTests(contoursInd)
    listTests = []
    if(len(contoursInd) == 4):
        for i,c in enumerate(contoursInd):
            test = pPe.getIndTest(testSquareBGR,c)
            listTests.append(test)         
    listTests = inA.resizeAll(listTests)
    listTestsBinary = []
    listTestsBinaryMask = []
    listTestsBinaryMaskEroDil = []
    for i, test in enumerate(listTests):
        #Checkpoint
        #cv2.imwrite(str(i) + 'indRespy.png', pP.BGR2gray(cv2.cvtColor(test,cv2.COLOR_RGB2BGR)))
        testBin = pP.contourBinarizationOtsu(test, 3, 3, 45, 3, Gs = 0, inverse = False, mean = True)
        #Checkpoint
        #cv2.imwrite(str(i) + 'indBinpy.png', testBin)
        listTestsBinary.append(testBin)
    #X-Markers binary with mask
    for i, test in enumerate(listTestsBinary):
        testBinMask = inA.andOperation(test, mask)
        #Checkpoint
        #cv2.imwrite(str(i) + 'indBinMaskpy.png', testBinMask)
        plt.subplot(321 + i),plt.imshow(test, 'gray')
        plt.show()
        listTestsBinaryMask.append(testBinMask)
    
    # erode and dilate blobs
    for i, test in enumerate(listTestsBinaryMask):
        test = inA.erosionDilation(test, 3)
        #Checkpoint
        #cv2.imwrite(str(i) + 'indEroDilpy.png', test)
        test = cv2.bitwise_not(test)
        #Checkpoint
        #cv2.imwrite(str(i) + 'indNot.png', test)
        plt.subplot(321 + i),plt.imshow(test, 'gray')
        plt.title('TestsBinaryMaskEroDil')
        plt.show()
        listTestsBinaryMaskEroDil.append(test)
        
    markerArea = {
            'ESAT6': 0,
            'CF': 0,
            'RV': 0,
            'Control':0
        }
    for i,img in enumerate(listTestsBinaryMaskEroDil):
        area = 0
        blobs = inA.blobDetect(img)
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
        #plt.subplot(321 + i),plt.imshow(inA.drawBlobs(img, blobs), 'gray')
        plt.show()
    df = df.append(markerArea, ignore_index=True)
#%%
ESATMean = df['ESAT6'].mean()
CFMean = df['CF'].mean()
RVMean = df['RV'].mean()
ControlMean = df['Control'].mean()
#%%
fileName = 'WhitePixelsAreas.xlsx'
df.to_excel(fileName)