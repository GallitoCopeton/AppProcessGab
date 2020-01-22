#%% 
import operator
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

try:
    os.chdir(os.path.join(os.getcwd(), 'Helper notebooks'))
    print(os.getcwd())
except:
    print(os.getcwd())
# Scripts para leer y procesar imagen
sys.path.insert(0, '../Golden Master (AS IS)')
import sorts as srt
import readImage
import preProcessing as pP
import perspective as pPe
import indAnalysis as inA
# %% [markdown]
# ## Golden Master Image Algorithm (AS IS)
0

# %%
batch = False
qr = '601170500100404'

# %% [markdown]
# ### Read Image

# %%
# Read from local file local = True, from db local = False, count = Test repeated
imgBGR = readImage.readImage(qr, local = False, ext='png',  count = 0)
if(isinstance(imgBGR, str)):
    print(imgBGR)
else:
    # show the original image
    if (not batch):
        plt.subplot(111), plt.imshow(cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB))
        plt.show()
# %% [markdown]
# ### Find Border of Biggest Square

# %%
if(isinstance(imgBGR, str)):
    print(imgBGR)
else:
    imgResized = pP.resizeImg(imgBGR, 728)

    imgBinary = pP.contourBinarization(
        imgBGR, 3, 7, 85, 2, inverse=True, mean=False)

    # Checkpoint
    #cv2.imwrite('BSbinpy.png', imgBinary)

externalSquare = pP.findTreeContours(imgBinary)
bgrCopy = imgBGR.copy()
cv2.drawContours(bgrCopy, [externalSquare[1]], -1, (0, 255, 0), 3)

# show contour image
# show resized and binary images
if (not batch):
    print("Length:  " + str(len(externalSquare)) +
          " type: " + str(type(externalSquare)))
    print(imgBGR.shape)
    plt.subplot(131), plt.imshow(cv2.cvtColor(bgrCopy, cv2.COLOR_BGR2RGB))
    plt.subplot(132), plt.imshow(cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB))
    plt.subplot(133), plt.imshow(imgBinary, 'gray')
    plt.show()

# %% [markdown]
# ### Validate Perspective for Biggest Square

# %%
# Order points to identify region of interest
# printPoints(externalSquare)
ext_index = 1
externalOrdSquare = srt.sortPointsContours(externalSquare)
imgValidated = pPe.perspectiveTransform(
    imgBGR, externalOrdSquare[ext_index], -5)

if(not batch):
    print(externalOrdSquare[1])
    plt.subplot(121), plt.imshow(cv2.cvtColor(
        imgBGR, cv2.COLOR_BGR2RGB)), plt.title('Input')
    plt.subplot(122), plt.imshow(
        cv2.cvtColor(imgValidated, cv2.COLOR_BGR2RGB)),
    plt.title('Output')
    plt.show()

# %% [markdown]
# ### Validate Perspective for "Test Square"

# %%
# Test Square lives inside Biggest Square and holds individual test's squares (a.k.a. X-Markers and XM Group)
# Do the actual perspective transform (previous one was for validating test device is there, and all margins align)
perspectiveBinary = pPe.perspectiveTransform(
    imgBinary, externalOrdSquare[ext_index], -5, True)
perspectiveBGR = pPe.perspectiveTransform(
    imgBGR, externalOrdSquare[ext_index], -5)
external = pP.findExternalContours(
    perspectiveBinary)
testSquareBinary = pPe.getTestSquare(perspectiveBinary, external, True)
testSquareBGR = pPe.getTestSquare(perspectiveBGR, external)

if (not batch):
    print(len(external))
    plt.subplot(121), plt.imshow(perspectiveBinary,
                                 'gray'), plt.title('Biggest Square Binary')
    plt.subplot(122), plt.imshow(cv2.cvtColor(testSquareBGR,
                                              cv2.COLOR_BGR2RGB)), plt.title('Test Square')
    plt.show()

# %% [markdown]
# ### Histogram Equalization for "Test Square"

# %%
testSquareBGR = pP.equalizeHistogram(testSquareBGR)

if (not batch):
    plt.subplot(122), plt.imshow(cv2.cvtColor(testSquareBGR,
                                              cv2.COLOR_BGR2RGB)), plt.title('Test Square')
    plt.show()

# %% [markdown]
# ### Validate Individual Tests (a.k.a. X-M Markers)

# %%
testSquareBGRCopy = testSquareBGR.copy()
height, width = testSquareBGR.shape[:2]
areaInd = (height*width/8) - 10
# Binarization is required again because lost of details on first binarization
contoursInd = pP.findTreeContours(pP.contourBinarization(
    testSquareBGR, 3, 7, 85, 2, mean=False), areaInd)
if(len(contoursInd) == 7):
    contoursInd = contoursInd[1:]
for c in contoursInd:
    cv2.drawContours(testSquareBGRCopy, [c], -1, (0, 255, 0), 3)

if (not batch):
    print("Approx ind area: " + str(areaInd))
    print("Number of X-Markers found: " + str(len(contoursInd)))
# Draw contour for ind tests (a.k.a. X-Markers)
    plt.subplot(121), plt.imshow(cv2.cvtColor(
        testSquareBGRCopy, cv2.COLOR_BGR2RGB))
    plt.show()

# %% [markdown]
# ### Crop Individual Test's Images (a.k.a. X-M Markers)

# %%
srt.sortTests(contoursInd)
listTests = []
if(len(contoursInd) == 6):
    for i, c in enumerate(contoursInd):
        test = pPe.getIndTest(testSquareBGR, c)
        listTests.append(test)
    if (not batch):
        for i, test in enumerate(listTests):
            plt.subplot(
                321 + i), plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
        plt.show()
    print(test.shape)

# %% [markdown]
# ### X-Marker Analysis (a.k.a. Sitios de prueba)

# %%
mask = inA.readMask()

if (not batch):
    plt.subplot(121), plt.imshow(mask, 'gray')
    plt.show()


# %%
# Fixed size 90x90
listTests = inA.resizeAll(listTests)
listKMeansTestsBina = []
listKMeansTests = []
listTestsBinary = []

# resized RGB
if (not batch):
    for i, test in enumerate(listTests):
        # Reshape image to have the R G B features (3) each as a column (-1)
        Z = test.reshape((-1, 3))
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
            (test.shape))
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
        imgBinaMaskInv = cv2.bitwise_not(imgBinaMask)
        imgBinaMaskInvOpen = cv2.morphologyEx(
            imgBinaMaskInv, cv2.MORPH_OPEN, kernel)
        listTestsBinary.append(imgBina)
        listKMeansTestsBina.append(imgBinaMaskInvOpen)
        listKMeansTests.append(kMeansReconstructedReshaped)
        plt.subplot(
            321 + i), plt.imshow(cv2.cvtColor(imgBinaMaskInvOpen, cv2.COLOR_BGR2RGB))
        # Checkpoint
        #cv2.imwrite(str(i) + 'indRespy.png', cv2.cvtColor(test,cv2.COLOR_RGB2BGR))
    plt.show()
if (not batch):
    for i, img in enumerate(listKMeansTests):
        plt.subplot(321 + i), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Checkpoint
        #cv2.imwrite(str(i) + 'indRespy.png', cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    plt.show()

# %%
# Blob detection
listBlobs = []
for i, img in enumerate(listKMeansTestsBina):
    imgCopy = img.copy()
    blobs = inA.blobDetect(img)
    listBlobs.append(blobs)
    if(not batch):
        if(operator.mod(i, 2) != 0):  # Print two by line
            print("Blobs detected: " + str(len(listBlobs[i-1])) +
                  " Blobs detected: " + str(len(listBlobs[i])))
        plt.subplot(321 + i), plt.imshow(inA.drawBlobs(imgCopy, blobs), 'gray')
plt.show()
# Test results using blobs
for i, img in enumerate(listTestsBinary):
    if(operator.mod(i, 2) != 0):  # Print two by line
        if(not batch):
            print("Result: " + inA.blobAnalysis(img, listBlobs[i-1]) +
                  " Area: " + str(round(inA.areaEstimation(listBlobs[i-1]), 2)) +
                  "\t Result: " + inA.blobAnalysis(img, listBlobs[i]) +
                  " Area: " + str(round(inA.areaEstimation(listBlobs[i]), 2)))


# %%
