# %%
import os

import numpy as np
import pandas as pd

from IF2.Crop import croppingProcess as cP
from IF2.Marker import markerProcess as mP
from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.ReadImage import readImage as rI
from IF2.Shows.showProcesses import showImage as show

picturesPath = '../assetsForTests/variationTesting/'
picturesFullPath = [picturesPath + name for name in os.listdir(
    picturesPath) if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg')]
pictures = [iO.resizeImg(rI.readLocal(path), 728) for path in picturesFullPath]
firstTen = pictures[:10]
secondTen = pictures[10:20]
thirdTen = pictures[20:]
# %%
markerNames = ['E6', 'CF', 'RV', 'CT']
features2Extract = [
                    'totalArea']

markerDfList = []
for picture in pictures:
    testArea = cP.getTestArea(picture)
    markers = cP.getMarkers(testArea)
    allMarkersFeatures = []
    for marker, name in zip(markers, markerNames):
        features = inA.extractFeatures(marker, features2Extract)
        featureValues = list(features.values())
        allMarkersFeatures += featureValues
    iterables = [markerNames, features2Extract]
    index = pd.MultiIndex.from_product(iterables, names=['name', 'data'])
    singleMarkerDf = pd.DataFrame(np.array(allMarkersFeatures).reshape(
        1, len(allMarkersFeatures)), columns=index)
    markerDfList.append(singleMarkerDf)
    show(testArea)
completeMarkerDf = pd.concat(markerDfList)
completeMarkerDfT = pd.concat(markerDfList).T
completeMarkerDfT.columns = np.repeat(np.arange(1,11),3)
corrE6 = completeMarkerDf['E6'].corr().T
corrCF = completeMarkerDf['CF'].corr().T
corrRV = completeMarkerDf['RV'].corr().T
corrCT = completeMarkerDf['CT'].corr().T
# %%
size = 3
splitTestDfs = [completeMarkerDf.iloc[i:i + size, :]
                for i in range(0, len(completeMarkerDf), size)]
customSplitDfs = []
for df in splitTestDfs:
    descriptionDf = df.describe().T
    percentage = descriptionDf['std'] / descriptionDf['mean']
    descriptionDf['%std'] = percentage
    customSplitDfs.append(descriptionDf)
fullTestsDescriptionsValues = pd.concat(
    [df.drop(['count', '25%', '50%', '75%'], axis=1) for df in customSplitDfs]).values
fullTestsDescriptions = pd.DataFrame(fullTestsDescriptionsValues)
#fullTestsDescriptions.to_excel('descriptions18-02_2.xlsx', index=False)
# completeMarkerDf.T.to_excel('complete_measurementsFinal4.xlsx')
