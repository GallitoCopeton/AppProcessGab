# %%
import random
import re
import os

import pymongo
import cv2
import numpy as np
from matplotlib import pyplot as plt

from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
from ImageProcessing import indAnalysis as inA

from ImageProcessing import colorTransformations as cT
from ImageProcessing import binarizations as bN
from AppProcess.CroppingProcess import croppingProcess as cP

import qrQuery
from machineLearningUtilities import modelPerformance as mP

# %%
URI = 'mongodb+srv://findOnlyReadUser:RojutuNHqy@clusterfinddemo-lwvvo.mongodb.net/datamap?retryWrites=true'
collectionNameImages = 'imagestotals'
dbNameImages = 'datamap'
collectionImages = qrQuery.getCollection(URI, dbNameImages, collectionNameImages)
regx = re.compile("^102")
query = {'count': 0}
#query = {'fileName': '102190300100155'}
# %%
images = rI.customQuery(collectionImages, query, limit=20, order=-1)
for image in images:
    try:
        testSite = cP.getNonEqTestSite(image['file'])
        markers = cP.getMarkers(testSite)
        for marker in markers:
#            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#            marker = cv2.filter2D(marker, -1, kernel)
            quadrants = inA.imageQuadrantSplit(marker)
            for quadrant in quadrants:
                quadrantBina = bN.adapBinaInverse(cT.BGR2gray(quadrant), 25, 5)
                _, contours, hierarchy = cv2.findContours(quadrantBina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea)
                mask = np.zeros(quadrant.shape, np.uint8)
                cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
                plt.imshow(mask)
                plt.show()
                ask = input('Guardar?')
                if int(ask) == 1:
                    askQuadrant = input('Qué cuadrante es? ')
                    plt.imsave('./quadrant{}.png'.format(askQuadrant), mask, )
    except Exception as e:
        print(e)
        continue
#%%
masks = [inA.readMask(mask, 45) for mask in os.listdir('./') if 'quadrant' in mask]
fullMask = inA.mergeQuadrants(masks)
plt.imsave('./mask.png', fullMask)