import appsProcess as aP
import sys
import os

import cv2

workingPath = os.getcwd()
scriptsPath = '../Golden Master (AS IS)'
os.chdir(scriptsPath)
os.chdir(workingPath)


def doFullProcess(image, figsize=9, save=True, folder='./'):
    folderName = folder+image['qr']+'/'
    testFull = cv2.cvtColor(image['image'], cv2.COLOR_BGR2RGB)
    testSite, testSiteEq = aP.getTestSite(testFull)
    markers, markersEq = aP.getMarkers(testSite, testSiteEq)
    markerClusterProcess = aP.markerClusterProcessing(markers)
    markerDirectProcess = aP.markerDirectProcessing(markersEq)
    aP.showBothProcesses(
        testSite, markers, markerClusterProcess, markerDirectProcess, figsize, save, folder)
