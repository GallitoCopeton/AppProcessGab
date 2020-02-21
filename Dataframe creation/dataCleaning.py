# %%
import datetime
import json
import re

import pandas as pd
import qrQuery
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from machineLearningUtilities import nnUtils as nnU

# %% Paths and filenames
tablesPath = '../Feature Tables'
tableFolder = 'DF Feb 18 12_02_10'
ext = '.xlsx'
fullTablePath = '/'.join([tablesPath, tableFolder, tableFolder+ext])
nnSavesFolder = '../Models/ANNs'
iteration = 0
iterationDict = {}
todayDatetime = datetime.datetime.now()
# %% Train and test set
df = pd.read_excel(fullTablePath)
df.dropna(inplace=True)
#%% Check if any data is missing
print(df.isnull().sum())
#%% Describe the df
print(df.describe().T)