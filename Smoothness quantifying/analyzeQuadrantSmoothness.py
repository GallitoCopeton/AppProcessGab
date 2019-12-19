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
closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
for k, image in enumerate(images):
    try:
        try:
            testSite = cP.getNonEqTestSite(image['file'])
        except:
            testSite = cP.getNonEqTestSite(image)
        plt.imshow(testSite)
        plt.show()
        markers = cP.getMarkers(testSite)
        processedMarkers = []
        edgesQuadrants = []
        for j, marker in enumerate(markers):
            kernel = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
            marker = pP.adapHistogramEq(marker, 8, (5, 5))
            marker = cv2.filter2D(marker, -1, kernel)
            quadrants = inA.imageQuadrantSplit(marker)
            processedQuadrants = []
            for i, quadrant in enumerate(quadrants):
                quadrantMask = inA.readMask('./quadrant{}.png'.format(str(i+1)), 45)
                quadrantMask = iO.applyTransformation(quadrantMask, closingKernel, cv2.MORPH_ERODE)
                quadrantMask = iO.applyTransformation(quadrantMask, closingKernel, cv2.MORPH_ERODE)
                maskedQuadrant = iO.andOperation(quadrant, quadrantMask)
                processedQuadrants.append(maskedQuadrant)
            processedMarker = inA.mergeQuadrants(processedQuadrants)
            edges = cv2.Canny(processedMarker,199, 200)
            edgesQuadrants.append(edges)
            processedMarkers.append(processedMarker)
        processedImage = inA.mergeQuadrants(processedMarkers)
        edgesImage = inA.mergeQuadrants(edgesQuadrants)
        plt.imshow(edgesImage)
        plt.show()
    except Exception as e:
        print(e)
        continue