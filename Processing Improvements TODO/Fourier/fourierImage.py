# %%
import datetime
import os
import re

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import qrQuery
from AppProcess.CroppingProcess import croppingProcess as cP

from AppProcess.MarkerProcess import markerProcess
from ImageProcessing import imageOperations as iO
from ImageProcessing import indAnalysis as inA
from ImageProcessing import colorTransformations as cT
from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
from ImageProcessing import binarizations as bZ
from ImageProcessing import preProcessing as pP


def fixMarker(marker): return iO.resizeFixed(rI.readb64(marker['image']))


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# %% Collections
realURI = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
realDbName = 'prodLaboratorio'
realCollectionName = 'markerTotals'
realMarkerCollection = qrQuery.getCollection(
    realURI, realDbName, realCollectionName)

cleanURI = 'mongodb://findUser:85d4s32D2%23diA@idenmon.zapto.org:888/?authSource=testerSrv'
cleanDbName = 'testerSrv'
cleanCollectionName = 'cleanMarkerTotals'
cleanCollection = qrQuery.getCollection(
    cleanURI, cleanDbName, cleanCollectionName)
# %%
# Info of the markers we want to analyze
# Mask creation
fourierMask = pP.median(bZ.simpleBinarization(cT.BGR2gray(
    iO.resizeFixed(rI.readLocal('./fourierImageMask5.png'))), 1), 3)
mask = bZ.otsuBinarize(cT.BGR2gray(
    rI.readLocal('../Smoothness quantifying/mask.png')))
markerNamesReal = ['ESAT6','RV1681','CFP10','P24']
features2Extract = ['nBlobs', 'totalArea', 'fullBlobs', 'bigBlobs', 'medBlobs',
                    'smallBlobs', 'q0HasBlob', 'q1HasBlob', 'q2HasBlob', 'q3HasBlob', 'diagnostic']
# Query: markers I want, that their diagnostic exists
limit = 37*4
markersP = realMarkerCollection.find(
    {'marker': {'$in': markerNamesReal}, 'diagnostic': 'P'}).limit(limit).sort('_id', -1)
markersN = realMarkerCollection.find(
    {'marker': {'$in': markerNamesReal}, 'diagnostic': 'N'}).limit(limit).sort('_id', -1)
print('Queries done')
#%%
localImagesPath = '../Zepto/Negativas'

localImagesNames = [iO.resizeImg(rI.readLocal('/'.join([localImagesPath, name])), 728) for name in os.listdir(localImagesPath) if name.endswith('.jpg')]
#%%
#mask = iO.applyTransformation(mask, np.ones((3, 3)), cv2.MORPH_ERODE, 1)
#for image in localImagesNames:
#    try:
#        testSite = cP.getTestArea(image)
#    except Exception as e:
#        print(e)
#        continue
#    try:
#        markers = cP.getMarkers(testSite)
#    except Exception as e:
#        print(e)
#        continue
#    for marker in markers:
#        markerNormLight = pP.normalizeLight(marker)
#        markerGray = cT.BGR2gray(markerNormLight)
#        markerDft = cv2.dft(np.float32(markerGray), flags=cv2.DFT_COMPLEX_OUTPUT)
#        markerDftShift = np.fft.fftshift(markerDft)
#        markerMS = 20 * \
#        np.log(cv2.magnitude(markerDftShift[:, :, 0], markerDftShift[:, :, 1]))
#        markerDftPMasked = iO.andOperation(markerDftShift, fourierMask)
#        markerIshift = np.fft.ifftshift(markerDftPMasked)
#        markerBack = cv2.idft(markerIshift)
#        markerBack = cv2.magnitude(markerBack[:, :, 0], markerBack[:, :, 1])
#        
#        markerBackMasked = iO.andOperation(markerBack, mask)
#        noise = cv2.subtract(np.float32(markerGray), markerBackMasked, dtype=cv2.CV_32F)
#       
#        sP.showImage(markerBackMasked, figSize=(3,3))
#        print(noise.sum())
        
    
# %%
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kColors = 3
attempts = 2
markersN = []
limit = 37*4
markersP = realMarkerCollection.find(
    {'marker': {'$in': markerNamesReal}, 'diagnostic': 'p'}).limit(limit).sort('_id', 1)
for image in localImagesNames:
    try:
        testSite = cP.getTestArea(image)
    except Exception as e:
        print(e)
        continue
    try:
        markers = cP.getMarkers(testSite)
    except Exception as e:
        print(e)
        continue
    [markersN.append(marker) for marker in markers]
#markersN = realMarkerCollection.find(
#    {'marker': {'$in': markerNamesReal}, 'diagnostic': 'N'}).limit(limit).sort('_id', -1)
for markerP, markerN in zip(markersP, markersN):
    print('*'*20)
    diagP = markerP['diagnostic']
#    diagN = markerN['diagnostic']
    markerP = fixMarker(markerP)
#    markerN = fixMarker(markerN)
    markerP = pP.normalizeLight(markerP)
    markerN = pP.normalizeLight(markerN)
    markerPGray = cT.BGR2gray(markerP)
    markerNGray = cT.BGR2gray(markerN)
    dftP = cv2.dft(np.float32(markerPGray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftN = cv2.dft(np.float32(markerNGray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftP_shift = np.fft.fftshift(dftP)
    dftN_shift = np.fft.fftshift(dftN)
    magnitude_spectrumP = 20 * \
        np.log(cv2.magnitude(dftP_shift[:, :, 0], dftP_shift[:, :, 1]))
    magnitude_spectrumN = 20 * \
        np.log(cv2.magnitude(dftN_shift[:, :, 0], dftN_shift[:, :, 1]))
    dftPMasked = iO.andOperation(dftP_shift, fourierMask)
    dftNMasked = iO.andOperation(dftN_shift, fourierMask)
    fP_ishift = np.fft.ifftshift(dftPMasked)
    fN_ishift = np.fft.ifftshift(dftNMasked)
    fPback = cv2.idft(fP_ishift)
    fNback = cv2.idft(fN_ishift)
    fPback = cv2.magnitude(fPback[:, :, 0], fPback[:, :, 1])
    fNback = cv2.magnitude(fNback[:, :, 0], fNback[:, :, 1])
    fPback = iO.andOperation(fPback, iO.applyTransformation(mask, np.ones((2, 2)), cv2.MORPH_ERODE, 2))
    fNback = iO.andOperation(fNback, iO.applyTransformation(mask, np.ones((2, 2)), cv2.MORPH_ERODE, 2))
    noiseP = cv2.subtract(np.float32(markerPGray), fPback, dtype=cv2.CV_32F)
    noiseN = cv2.subtract(np.float32(markerNGray), fNback)
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=False)
    axP = axs[0][0]
    axN = axs[1][0]
    spectrumP = axs[0][1]
    spectrumN = axs[1][1]
    axP.set_title('P')
    axN.set_title('N')
    spectrumP.set_title('spectrum')
    spectrumN.set_title('spectrum')
    axP.imshow(cT.BGR2RGB(markerP))
    axN.imshow(cT.BGR2RGB(markerN))
    spectrumP.imshow(fPback)
    spectrumN.imshow(fNback)
    plt.show()
    print(noiseP.sum())
    print(noiseN.sum())
    print('*'*20)
