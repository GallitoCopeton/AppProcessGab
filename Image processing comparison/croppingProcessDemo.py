import math
import json
import os
from bson import ObjectId

import numpy as np

import qrQuery
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.Processing import colorTransformations as cT
from machineLearningUtilities import modelPerformance  as moPe
from IF2.Shows.showProcesses import showImage as show
from IF2.ReadImage import readImage as rI
from IF2.Marker import markerProcess as mP
from IF2.Crop import croppingProcess as cP

with open('../Database connections/connections.json') as jsonFile:
    connections = json.load(jsonFile)['connections']
#%% Validation database
zeptoConnection = connections['testingZapto']
zaptoImagesCollection = qrQuery.getCollection(
zeptoConnection['URI'], zeptoConnection['databaseName'], zeptoConnection['collections']['markersCollectionName'])
#%% Get image
marker = zaptoImagesCollection.find_one({'diagnostic': 'p'})
markersImage = cT.BGR2RGB(iO.resizeFixed(rI.readb64(marker['image'])))
markersInfo = {'diagnostic': marker['diagnostic'],
                 'name':  marker['marker'],
                 'qr': marker['QR'],
                 'count': marker['count'],
                 '_id': marker['_id']}
#%% Start process
originalFig = show(markersImage, title='Marcador original', returnFig=True)
binarizedMarker = mP.clusteringProcess([markersImage], 3, 6)[0]
binarizedFig = show(binarizedMarker, title='Marcador binarizado', returnFig=True)
# Get the total area per quadrant
totalArea = inA.extractFeatures(markersImage, ['totalArea'])['totalArea']
#%% Save the figures to the assets folder
assetsFolderPath = '../assetsForTests'
figuresFolder = 'Marker Processes'
figuresPath = '/'.join([assetsFolderPath, figuresFolder])
qrQuery.makeFolders(figuresPath)
originalFigName = '/Original Marker'
binarizedFigName = '/Binarized Marker'
originalFig.savefig(figuresPath+originalFigName)
binarizedFig.savefig(figuresPath+binarizedFigName)