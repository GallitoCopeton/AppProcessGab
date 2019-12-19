# %%
import re
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ReadImages import readImage as rI
from ImageProcessing import indAnalysis as inA
from ImageProcessing import colorTransformations as cT
from ImageProcessing import binarizations as bN
from ImageProcessing import preProcessing as pP
from AppProcess.CroppingProcess import croppingProcess as cP
from ShowProcess import showProcesses as sP
from ImageProcessing import imageOperations as iO

import qrQuery

# %%
URI = 'mongodb+srv://findOnlyReadUser:RojutuNHqy@clusterfinddemo-lwvvo.mongodb.net/datamap?retryWrites=true'
collectionNameImages = 'imagestotals'
dbNameImages = 'datamap'
collectionImages = qrQuery.getCollection(URI, dbNameImages, collectionNameImages)
regx = re.compile("^102")
query = {'count': 0}
query = {'fileName': {'$in': ['102190800200474', '102190800200130', '102190900100123', '102190900100005']}, 'count': 0}
# %%
cloudImages = rI.customQuery(collectionImages, query, limit=10, order=-1)
localImages = [rI.readLocal(image) for image in os.listdir('./') if image.endswith('.jpeg') or image == '0.png']
images = cloudImages + localImages
erodingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
for k, image in enumerate(images):
    try:
        try:
            testSite = cP.getNonEqTestSite(image['file'])
        except:
            testSite = cP.getNonEqTestSite(image)
        sP.showImage(testSite, 'original test site', figSize=(5, 5))
        markers = cP.getMarkers(testSite)
        processedMarkers = []
        edgesQuadrants = []
        for j, marker in enumerate(markers):
            kernel = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
            marker = pP.adapHistogramEq(marker, 8, (5, 5))
            marker = cv2.filter2D(marker, -1, kernel)
            markerMask = inA.readMask('./mask.png')
            markerMask = iO.applyTransformation(markerMask, erodingKernel, cv2.MORPH_ERODE)
            markerMask = iO.applyTransformation(markerMask, erodingKernel, cv2.MORPH_ERODE)
            processedMarker = iO.andOperation(marker, markerMask)
            edges = cv2.Canny(processedMarker,199, 200)
            edgesQuadrants.append(edges)
            processedMarkers.append(processedMarker)
        processedImage = inA.mergeQuadrants(processedMarkers)
        edges = cv2.Canny(processedImage,199, 200)
        sP.showImage(edges, 'Gradient image', figSize=( 5,5))
    except Exception as e:
        print(e)
        continue# -*- coding: utf-8 -*-

