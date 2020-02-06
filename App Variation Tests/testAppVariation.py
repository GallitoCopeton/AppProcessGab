# %%
import os

from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.ReadImage import readImage as rI
from IF2.Crop import croppingProcess as cP
from IF2.Marker import markerProcess as mP
from IF2.Shows.showProcesses import showImage as show




picturesPath = '../assetsForTests/variationTesting/'
picturesFullPath = [picturesPath+name for name in os.listdir(picturesPath) if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg')]
pictures = [iO.resizeImg(rI.readLocal(path), 728) for path in picturesFullPath]
firstTen = pictures[:10]
secondTen = pictures[10:20]
thirdTen = pictures[20:]
# %%
markerNames = ['E6','CF','RV','CT']
features2Extract = ['agl',
                    'aglMean',
                    'totalArea',
                    'fullBlobs',
                    'bigBlobs',
                    'medBlobs',
                    'smallBlobs',]
for picture in thirdTen:
    testArea = cP.getTestArea(picture)
    markers = cP.getMarkers(testArea)
    for marker, name in zip(markers, markerNames):
        features = inA.extractFeatures(marker, features2Extract)
        print(f'Result for marker {name}')
        for feature in features.keys():
            featureRes = features[feature]
            print(f'Result for feature {feature}: {featureRes}')
    show(testArea, figSize=(5,5))