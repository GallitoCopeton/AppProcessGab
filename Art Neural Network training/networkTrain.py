# %%
import datetime
import json
import re

import pandas as pd
import qrQuery
from sklearn.preprocessing import StandardScaler

import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from machineLearningUtilities import nnUtils as nnU

# %% Paths and filenames
tablesPath = '../Feature Tables'
tableFolder = 'DF Jan 24 16_47_20'
ext = '.xlsx'
fullTablePath = '/'.join([tablesPath, tableFolder, tableFolder+ext])
nnSavesFolder = '../Models/ANNs'
# %% Train and test set
df = pd.read_excel(fullTablePath)
df.dropna(inplace=True)
X = mlU.getFeatures(df, 0, -1)
y = mlU.getLabels(df, 'diagnostic')
# %% Split data
X_train, X_test, y_train, y_test = mlU.splitData(X, y, .3)
# %% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
means = sc.mean_
variances = sc.var_
# %% Baseline model
todayDatetime = datetime.datetime.now()
alpha = 3
nFeatures = X.shape[1]
outputNeurons = 1
nSamples = len(X_train)
activations = ['relu', 'relu', 'sigmoid']
l1 = 0.01
dropout = 0.0
batchNorm = False
model = nnU.createANN(alpha=alpha, features=nFeatures, outputNeurons=outputNeurons, nSamples=nSamples,
                      activations=activations, l1=l1, dropout=dropout, batchNorm=batchNorm)
model.compile(optimizer='nadam', loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])
modelHistory = model.fit(X_train, y_train, batch_size=300,
                         epochs=1600, verbose=2, validation_data=(X_test, y_test))
nnU.plot_history([('Base model', modelHistory)])
yPred = nnU.performance(model, X_test, y_test)
# %%
dateString = re.sub(r':', '_', todayDatetime.ctime())[4:-5]
currentNNFolder = f'ANN_{yPred[0].round(2)} date {dateString}'
nnFilename = currentNNFolder+'.pkl'
currentNNPath = '/'.join([nnSavesFolder, currentNNFolder])
qrQuery.makeFolders(currentNNPath)
nnInfoJsonFileName = 'nnInfo.json'
nnInfoJsonFilePath = '/'.join([currentNNPath, nnInfoJsonFileName])
joinedMeans = ','.join([str(mean) for mean in means])
joinedVars = ','.join([str(var) for var in variances])
joinedFeatures = ','.join(X.columns)
outputFeatures = {
            'means': joinedMeans,
            'variances': joinedVars,
            'features': joinedFeatures
        }
with open(nnInfoJsonFilePath, 'w') as jsonOut:
    json.dump(outputFeatures, jsonOut)
mP.saveModel(currentNNPath, nnFilename, model)
