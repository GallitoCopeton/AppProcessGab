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
tableFolder = 'DF Feb  4 13_48_03'
ext = '.xlsx'
fullTablePath = '/'.join([tablesPath, tableFolder, tableFolder+ext])
nnSavesFolder = '../Models/ANNs'
iteration = 0
iterationDict = {}
todayDatetime = datetime.datetime.now()
# %% Train and test set
df = pd.read_excel(fullTablePath)
df.dropna(inplace=True)
sns.countplot(x='diagnostic', data=df)
X = mlU.getFeatures(df, 0, -1)
y = mlU.getLabels(df, 'diagnostic')
# %% Split data
split = .2
seed = np.random.randint(0, 500)
X_train, X_test, y_train, y_test = mlU.splitData(X, y, split, seed=seed)
# %% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
means = sc.mean_
variances = sc.var_
# %% Baseline model
dateString = re.sub(r':', '_', todayDatetime.ctime())[4:-5]
currentNNFolder = f'ANN_date {dateString}'
currentNNPath = '/'.join([nnSavesFolder, currentNNFolder])
qrQuery.makeFolders(currentNNPath)
alpha = 5
nFeatures = X.shape[1]
outputNeurons = 1
nSamples = len(X_train)
activations = ['sigmoid', 'relu', 'relu']
l1 = 0.01
l2 = None
dropout = 0.5
batchNorm = False
epochs = 500
batch_size = 2**6
optimizer = 'adam'
loss = 'binary_crossentropy'
model = nnU.createANN(alpha=alpha, features=nFeatures, outputNeurons=outputNeurons, nSamples=nSamples,
                      activations=activations, l1=l1, l2=l2, dropout=dropout, batchNorm=batchNorm)
model.compile(optimizer=optimizer, loss=loss,
              metrics=['accuracy', 'binary_crossentropy'])
modelHistory = model.fit(X_train, y_train, batch_size=batch_size,
                         epochs=epochs, verbose=2, validation_data=(X_test, y_test))
nnU.plot_history([('Base model', modelHistory)])
yPred = nnU.performance(model, X_test, y_test)
nnInfoJsonFileName = 'nnInfo.json'
nnInfoJsonFilePath = '/'.join([currentNNPath, nnInfoJsonFileName])
joinedMeans = ','.join([str(mean) for mean in means])
joinedVars = ','.join([str(var) for var in variances])
joinedFeatures = ','.join(X.columns)
nnFilename = str(int(datetime.datetime.timestamp(datetime.datetime.now())))+'.pkl'

outputFeatures = {
            'means': joinedMeans,
            'variances': joinedVars,
            'features': joinedFeatures,
            'params': {
                    'alpha': alpha,
                    'l1': l1,
                    'l2': l2,
                    'dropout': dropout,
                    'batchNorm': batchNorm,
                    'acc': yPred[0],
                    'epochs': epochs,
                    'batchSize': batch_size,
                    'split': split,
                    'seed': seed,
                    'activations': ', '.join(activations),
                    'optimizer': optimizer,
                    'loss': loss,
                    'trainSamples': nSamples,
                    'nSamples': len(df),
                    'fileName': nnFilename,
                    'trainingFileName': tableFolder
                    }
        }
iterationDict[str(iteration)] = outputFeatures
with open(nnInfoJsonFilePath, 'w') as jsonOut:
    json.dump(iterationDict, jsonOut)

mP.saveModel(currentNNPath, nnFilename, model)
iteration += 1
# %%

