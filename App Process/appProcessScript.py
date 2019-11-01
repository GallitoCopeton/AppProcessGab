import sys
import os

import cv2

workingPath = os.getcwd()
scriptsPath = '../Golden Master (AS IS)'
os.chdir(scriptsPath)
import appProcess as aP
os.chdir(workingPath)
# %%
#imagesPath = './Positives/'
imagesPath = '../Image processing comparison/Errors/'
images = os.listdir(imagesPath)

for image in images:
    filePath = os.path.join(imagesPath, image)
    testFull = cv2.imread(filePath)
    try:
        testSite, testSiteEq = aP.getTestSite(testFull)
    except:
        continue
    try:
        markers, markersEq = aP.getMarkers(testSite, testSiteEq)
    except:
        continue
    markerClusterProcess = aP.markerClusterProcessing(markers)
    markerDirectProcess = aP.markerDirectProcessing(markersEq)
    aP.showBothProcesses(
        testSite, markers, markerClusterProcess, markerDirectProcess)
