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
from ImageProcessing import imageOperations as iO
import qrQuery
from machineLearningUtilities import modelPerformance as mP
# %%
URI = 'mongodb+srv://findOnlyReadUser:RojutuNHqy@clusterfinddemo-lwvvo.mongodb.net/datamap?retryWrites=true'
collectionNameImages = 'imagestotals'
dbNameImages = 'datamap'
collectionImages = qrQuery.getCollection(URI, dbNameImages, collectionNameImages)
regx = re.compile("^102")
query = {'count': 0}
query = {'fileName': '102190300100155'}
#%%
scriptPath = re.sub(r'\\', '/', os.path.dirname(os.path.realpath(__file__)))
filePath = '/'.join([scriptPath, '../ML algorithms/models'])
fileName = 'RandomForestClassifier.pkl'
model = mP.loadModel(filePath, fileName)
# %%

images = rI.customQuery(collectionImages, query, limit=6, order=-1)
for image in images:
    m = sP.showClusterProcess(image['file'], 3, 20, (8, 9), True)
    if m:
        for marker in m[3]:
            values = np.array(list(inA.deepQuadrantAnalysis(iO.notOperation(marker)).values())).reshape(1, -1)
        
