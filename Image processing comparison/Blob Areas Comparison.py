
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Scripts para leer y procesar imagen
workingPath = os.getcwd()
scriptsPath = '../Golden Master (AS IS)'
os.chdir(scriptsPath)
import indAnalysis as inA
import perspective as pPe
import preProcessing as pP
import sorts as srt
os.chdir(workingPath)
# %%
#imagesPath = './Positives/'
imagesPath = './Errors/'
images = os.listdir(imagesPath)
mask = inA.readMask()

# %%
df = pd.DataFrame()
columns = ['ESAT6', 'CF', 'RV', 'Control', 'file name']
originalImages = []
cards = []
testSites = []
testSitesEq = []
# GET TEST SITES
for imageNumber, imageName in enumerate(images):
    filePath = os.path.join(imagesPath, imageName)
    testFull = cv2.imread(filePath)
    testResized = pP.resizeImg(testFull, 728)
    testBin = pP.contourBinarization(
        testResized, 3, 7, 85, 2, inverse=True, mean=False)
    cardContours = pP.findTreeContours(testBin)
    for contour in cardContours:
        orderedContour = srt.sortPoints(contour)
        cardBin = pPe.perspectiveTransform(
            testBin, orderedContour, -5, binary=True)
        qrAndTestSiteContours = pP.findExternalContours(cardBin)
        if len(qrAndTestSiteContours) == 2:
            card = pPe.perspectiveTransform(
                testResized, orderedContour, -5)
            contour1, contour2 = qrAndTestSiteContours
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            if area1 > area2:
                testSiteContours = contour1
                qrSiteContours = contour2
            else:
                testSiteContours = contour2
                qrSiteContours = contour1
            testSiteContoursOrdered = srt.sortPoints(testSiteContours)
            qrSiteContoursOrdered = srt.sortPoints(qrSiteContours)
            if qrSiteContoursOrdered[0][0][0] > testSiteContoursOrdered[0][0][0] and qrSiteContoursOrdered[2][0][1] > testSiteContoursOrdered[2][0][1]:
                testSite = pPe.perspectiveTransform(
                    card, testSiteContoursOrdered, offset=5)
                testSiteEq = pP.equalizeHistogram(testSite)
                testSites.append(testSite)
                testSitesEq.append(testSiteEq)
            else:
                print('Could not get test site this time')
                continue
    if testSite is None:
        print('Error while getting test site')
        continue

# GET MARKERS FOR CLUSTER AND DIRECT BINARIZATION
jList = np.arange(0, len(images), 1)
nthImage = 0
for (j, testSite, testSiteEq) in zip(jList, testSites, testSitesEq):
    print(f'Name of file: {images[j]}')
    height, width = testSite.shape[:2]
    markersContoursEq = pP.findTreeContours(pP.contourBinarization(
        testSiteEq, 3, 7, 85, 2, mean=False), 115000)
    if len(markersContoursEq) == 5 or len(markersContoursEq) == 7:
        markersContoursEq = markersContoursEq[1:]
    markersEq = []
    markers = []
    if(len(markersContoursEq) == 4 or len(markersContoursEq) == 6):
        srt.sortTests(markersContoursEq)
        for i, markerContour in enumerate(markersContoursEq):
            # Equalizado
            markerEq = pPe.getIndTest(testSiteEq, markerContour)
            markersEq.append(markerEq)
            # No equalizado
            marker = pPe.getIndTest(testSite, markerContour)
            markers.append(marker)
    markersEq = inA.resizeAll(markersEq)
    markers = inA.resizeAll(markers)
    clustersNot = []
    directNot = []
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 1000, 100)
    k = 3
    attempts = 80
    iList = np.arange(0, len(markers), 1)
    for (marker, markerEq, i) in zip(markers, markersEq, iList):
        # CLUSTER (Loki) ==> START
        clusterRecon = pP.clusterReconstruction(marker, criteria, k, attempts)
        clusterGray = cv2.cvtColor(clusterRecon, cv2.COLOR_BGR2GRAY)
        _, clusterBin = cv2.threshold(clusterGray, 150, 255, cv2.THRESH_BINARY)
        clusterMasked = inA.andOperation(clusterBin, mask)
        dilateKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        openKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clusterTrans = cv2.morphologyEx(
            clusterMasked, cv2.MORPH_DILATE, dilateKernel)
        clusterTrans = cv2.morphologyEx(
            clusterTrans, cv2.MORPH_OPEN, openKernel)
        clusterNot = cv2.bitwise_not(clusterTrans)
        # CLUSTER (Loki) ==> FINISH
        # PRODUCTION AREAS (Xplora) ==> START
        markerBin = pP.contourBinarizationOtsu(
            markerEq, 3, 3, 45, 3, Gs=0, inverse=False, mean=True)
        markerMasked = inA.andOperation(markerBin, mask)
        markerTrans = inA.erosionDilation(markerMasked, 3)
        markerNot = cv2.bitwise_not(markerTrans)
        # PRODUCTION AREAS (Xplora) ==> FINISH
        # Appends
        clustersNot.append(clusterNot)
        directNot.append(markerNot)

        # PLOTS
        fig = plt.figure(figsize=(4, 4), constrained_layout=True)
        gs = fig.add_gridspec(6, 4)
        gridSpecTupple = gs.get_geometry()
        halfHeight = int(gridSpecTupple[0]/2)
        halfWidth = int(gridSpecTupple[1]/2)
        fig.set_facecolor((0.6, 0.6, 0.6))

        plotFullCard = fig.add_subplot(gs[0:halfHeight+1, 0:halfWidth])
        plotMarker = fig.add_subplot(gs[0:halfHeight, 1-halfHeight:])
        plotClusterRecon = fig.add_subplot(gs[4, 0])
        plotClusterBin = fig.add_subplot(gs[4, 1])
        plotClusterTrans = fig.add_subplot(gs[4, 2])
        plotClusterBlobs = fig.add_subplot(gs[4, 3])
        plotMarkerEq = fig.add_subplot(gs[5, 0])
        plotMarkerBin = fig.add_subplot(gs[5, 1])
        plotMarkerTrans = fig.add_subplot(gs[5, 2])
        plotMarkerBlobs = fig.add_subplot(gs[5, 3])

        plotFullCard.set_title('Original image')
        plotMarker.set_title(f'Marker {i+1}')
        plotClusterRecon.set_title('Reconstruction')
        plotClusterBin.set_title('Binarized')
        plotClusterTrans.set_title('Morph')
        plotClusterBlobs.set_title('Blobs')
        plotMarkerEq.set_title('Equalization')
        plotMarkerBin.set_title('Binarized')
        plotMarkerTrans.set_title('Morph')
        plotMarkerBlobs.set_title('Blobs')

        plotFullCard.set_axis_off()
        plotMarker.set_axis_off()
        plotClusterRecon.set_axis_off()
        plotClusterBin.set_axis_off()
        plotClusterTrans.set_axis_off()
        plotClusterBlobs.set_axis_off()
        plotMarkerEq.set_axis_off()
        plotMarkerBin.set_axis_off()
        plotMarkerTrans.set_axis_off()
        plotMarkerBlobs.set_axis_off()

        plotFullCard.imshow(testSites[j])
        plotMarker.imshow(marker)
        plotClusterRecon.imshow(clusterRecon)
        plotClusterBin.imshow(clusterBin, 'gray')
        plotClusterTrans.imshow(clusterTrans, 'gray')
        plotClusterBlobs.imshow(clusterNot, 'gray')
        plotMarkerEq.imshow(markerEq)
        plotMarkerBin.imshow(markerBin, 'gray')
        plotMarkerTrans.imshow(markerTrans, 'gray')
        plotMarkerBlobs.imshow(markerNot, 'gray')
        plt.show()
        plt.close(fig)
# %%
    nMarkers = len(markers)
    iList = np.arange(0, nMarkers)
    if nMarkers == 4:
        results = {
            'fileName': '',
            'clusters': {
                'ESAT6': 0,
                'CF': 0,
                'RV': 0,
                'Control': 0
            },
            'direct': {
                'ESAT6': 0,
                'CF': 0,
                'RV': 0,
                'Control': 0
            }
        }
        for clusterNot, markerNot, i in zip(clustersNot, directNot, iList):

            clusterNot_area = 0
            markerNot_area = 0

            clusterNot_blobs = inA.blobDetect(clusterNot)
            markerNot_blobs = inA.blobDetect(markerNot)

            for clusterNot_blob in clusterNot_blobs:
                clusterNot_area += clusterNot_blob.size
            if i == 0:
                results['clusters']['ESAT6'] += clusterNot_area
            elif i == 1:
                results['clusters']['CF'] += clusterNot_area
            elif i == 2:
                results['clusters']['RV'] += clusterNot_area
            elif i == 3:
                results['clusters']['Control'] += clusterNot_area

            for markerNot_blob in (markerNot_blobs):
                markerNot_area += markerNot_blob.size
            if i == 0:
                results['direct']['ESAT6'] += markerNot_area
            elif i == 1:
                results['direct']['CF'] += markerNot_area
            elif i == 2:
                results['direct']['RV'] += markerNot_area
            elif i == 3:
                results['direct']['Control'] += markerNot_area
        results['fileName'] = imageName[-10:]
    resultsDf = pd.DataFrame.from_records([results], columns=columns)
    df = df.append(resultsDf)
    if nMarkers == 6:
        results = {
            'fileName': '',
            'clusters': {
                'ESAT6': 0,
                'CF': 0,
                'RV': 0,
                'P24': 0,
                'P26': 0,
                'Control': 0
            },
            'direct': {
                'ESAT6': 0,
                'CF': 0,
                'RV': 0,
                'P24': 0,
                'P26': 0,
                'Control': 0
            }
        }
        for clusterNot, markerNot, i in zip(clustersNot, directNot, iList):

            clusterNot_area = 0
            markerNot_area = 0

            clusterNot_blobs = inA.blobDetect(clusterNot)
            markerNot_blobs = inA.blobDetect(markerNot)

            for clusterNot_blob in clusterNot_blobs:
                clusterNot_area += clusterNot_blob.size
            if i == 0:
                results['clusters']['ESAT6'] += clusterNot_area
            elif i == 1:
                results['clusters']['CF'] += clusterNot_area
            elif i == 2:
                results['clusters']['RV'] += clusterNot_area
            elif i == 3:
                results['clusters']['P24'] += clusterNot_area
            elif i == 4:
                results['clusters']['P26'] += clusterNot_area
            elif i == 5:
                results['clusters']['Control'] += clusterNot_area

            for markerNot_blob in (markerNot_blobs):
                markerNot_area += markerNot_blob.size
            if i == 0:
                results['direct']['ESAT6'] += markerNot_area
            elif i == 1:
                results['direct']['CF'] += markerNot_area
            elif i == 2:
                results['direct']['RV'] += markerNot_area
            elif i == 3:
                results['direct']['P24'] += markerNot_area
            elif i == 4:
                results['direct']['P26'] += markerNot_area
            elif i == 5:
                results['direct']['Control'] += markerNot_area
