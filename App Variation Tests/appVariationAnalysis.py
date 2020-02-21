#%%
import os

import pandas as pd
import numpy as np

from IF2.Processing import imageOperations as iO
from IF2.Processing import indAnalysis as inA
from IF2.ReadImage import readImage as rI
from IF2.Crop import croppingProcess as cP
from IF2.Marker import markerProcess as mP
from IF2.Shows.showProcesses import showImage as show
#%%
features = ['agl',
                    'aglMean',
                    'totalArea',
                    'bigBlobs',
                    'medBlobs']
markerNames = ['E6','CF','RV','CT']
iterables = [markerNames, features]
index = pd.MultiIndex.from_product(iterables, names=['name', 'data'])
PATH_TO_EXCEL = './app_measurements.xlsx'
appMeasurementsDf = pd.read_excel(PATH_TO_EXCEL).T

appMeasurementsDf.columns = index
#%%
size = 3
splitTestDfs = [appMeasurementsDf.iloc[i:i+size,:] for i in range(0, len(appMeasurementsDf),size)]
customSplitDfs = []
for df in splitTestDfs:
    descriptionDf = df.describe().T
    percentage = descriptionDf['std']/descriptionDf['mean']
    descriptionDf['%std'] = percentage
    customSplitDfs.append(descriptionDf)
fullTestsDescriptions = pd.concat([df.drop(['count','25%','50%','75%'], axis=1) for df in customSplitDfs])
copyApp = fullTestsDescriptions.values
#fullTestsDescriptions.to_excel('appDescriptions.xlsx')
#%%
features = ['agl',
                    'aglMean',
                    'totalArea',
                    'bigBlobs',
                    'medBlobs']
markerNames = ['E6','CF','RV','CT']*10
iterables = [markerNames, features]
index = pd.MultiIndex.from_product(iterables, names=['name', 'data'])
PATH_TO_PYTHON_DESCRIPTION_EXCEL = './descriptions18-02_2.xlsx'
pythonDescriptions = pd.read_excel(PATH_TO_PYTHON_DESCRIPTION_EXCEL)
copyPython = pythonDescriptions.values
diff = pd.DataFrame(copyApp - copyPython)
diff.set_index(index, inplace=True)
diff.to_excel('differences18-02_2.xlsx')