# %%
import datetime
import json
import re

import pandas as pd
import qrQuery
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import seaborn as sns

import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from machineLearningUtilities import nnUtils as nnU

# %% Paths and filenames
tablesPath = '../Feature Tables'
tableFolder = 'DF Feb 18 12_02_10'
ext = '.xlsx'
fullTablePath = '/'.join([tablesPath, tableFolder, tableFolder+'_clean'+ext])
nnSavesFolder = '../Models/ANNs'
iteration = 0
iterationDict = {}
todayDatetime = datetime.datetime.now()
# %% Train and test set
df = pd.read_excel('marker1d.xlsx')
df.dropna(inplace=True)
#sns.countplot(x='diagnostic', data=df)
#%%
X = df.iloc[:,:-2].values
#X.drop('totalArea', inplace=True, axis=1)
#X.drop('agl', inplace=True, axis=1)
#X.drop('fullBlobs', inplace=True, axis=1)
#X.drop('smallBlobs', inplace=True, axis=1)
y = df.iloc[:,-1].values
# %% Split data
split = .2
seed = np.random.randint(0, 500)
X_train, X_test, y_train, y_test = mlU.splitData(X, y, split, seed=seed)
# %% Feature Scaling
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#means = sc.mean_
#variances = sc.var_
# %% Feature Scaling
mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
xMax = mm.data_max_
xMin = mm.data_min_

# %% Baseline model
dateString = re.sub(r':', '_', todayDatetime.ctime())[4:-5]
currentNNFolder = f'ANN_date {dateString}'
currentNNPath = '/'.join([nnSavesFolder, currentNNFolder])
qrQuery.makeFolders(currentNNPath)
alpha = 7
nFeatures = X.shape[1]
outputNeurons = 1
nSamples = len(X_train)
activations = ['relu', 'relu', 'relu', 'relu']
l1 = None
l2 = 0.001
dropout = 0.3
batchNorm = False
epochs = 500
batch_size = 2**10
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
joinedMax = ','.join([str(xmax) for xmax in xMax])
joinedMin = ','.join([str(xmin) for xmin in xMin])
joinedFeatures = ','.join(X.columns)
nnFilename = str(int(datetime.datetime.timestamp(datetime.datetime.now())))+'.pkl'

outputFeatures = {
            'max': joinedMax,
            'min': joinedMin,
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
from sklearn.model_selection import KFold
import numpy as np
fold = 1
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

kFold = KFold(n_splits=3, shuffle=True)
for train, test in kFold.split(inputs, targets):
    model = nnU.createANN(alpha=alpha, features=nFeatures, outputNeurons=outputNeurons, nSamples=nSamples,
                      activations=activations, l1=l1, l2=l2, dropout=dropout, batchNorm=batchNorm)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', 'binary_crossentropy'])
    print('-'*70)
    print(f'Training for fold {fold}')
    modelHistory = model.fit(inputs[train], targets[train], batch_size=batch_size,
                             epochs=epochs, verbose=2, validation_split=.2)
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    fold = fold + 1