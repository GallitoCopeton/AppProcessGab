# %%
import datetime
import os
import re

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import qrQuery
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.Processing import colorTransformations as cT
from IF2.Processing import binarizations as bZ
from IF2.Processing import preProcessing as pP
from IF2.Crop import croppingProcess as cP
from IF2.Marker import markerProcess as mP
from IF2.ReadImage import readImage as rI
from IF2.Shows.showProcesses import showImage as sI
from IF2.masks import importFourierMask, importFullMask, importMaskInv

def fixMarker(marker): return iO.resizeFixed(rI.readb64(marker['image']))
# %% Collections
realURI = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
realDbName = 'prodLaboratorio'
realCollectionName = 'markerTotals'
realMarkerCollection = qrQuery.getCollection(
    realURI, realDbName, realCollectionName)
# %%
# Mask imports
fourierMask = importFourierMask()
fullMask = importFullMask()
k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Query...
limit = 0
diagQ = {'$ne': None}
#diagQ = 'P'
markers = realMarkerCollection.find({'diagnostic': diagQ}).limit(limit)
markersInfo = [[(iO.resizeFixed(rI.readb64(marker['image']))),
                {'diagnostic': marker['diagnostic'],
                 'name':  marker['marker'],
                 'qr': marker['QR'],
                 'count': marker['count']}
                ] for marker in markers]
markerImages = [info[0] for info in markersInfo]
markersInfo = [info[1] for info in markersInfo]
# Local...
#localImagesPath = '../../assetsForTests/Negatives'
#localImagesNames = [iO.resizeImg(rI.readLocal('/'.join([localImagesPath, name])), 728) for name in os.listdir(localImagesPath) if name.endswith('.jpg')]

#%%
fullFeatures = []
for i, (markerImage, markerInfo) in enumerate(zip(markerImages, markersInfo)):
    diag = 1 if markerInfo['diagnostic'] == 'P' else 0
#    
    fourierMarker = mP.fourierProcess([markerImage])[0]
    noise = fourierMarker.sum()
    laplacian = cv2.Laplacian(fourierMarker, cv2.CV_32F)
    lapGrad = np.gradient(laplacian)
    gradSumX = lapGrad[0].sum()
    gradSumY = lapGrad[1].sum()
    markerMasked = mP.getBloodOnlyMask(markerImage)
    agl = cv2.Canny(markerMasked, 900, 900)
    sI(agl, title=diag)
    bloodAgl = agl.sum()
    features = ['totalArea']
    totalArea = inA.extractFeatures(markerImage, features)['totalArea']
    ratio = (abs(gradSumX)+abs(gradSumY))
    
    cols = ['agl', 'totalArea', 'ratio', 'noise', 'diag']
    features = [bloodAgl, totalArea, ratio, noise, diag]
    fullFeatures.append(features)
df = pd.DataFrame(fullFeatures, columns=cols)