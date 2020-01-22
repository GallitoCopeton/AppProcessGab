# %%
import datetime
import re

import pandas as pd
import qrQuery
from sklearn.preprocessing import StandardScaler

import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from machineLearningUtilities import nnUtils as nnU

# %% Paths and filenames
tablePath = '../Feature Tables'
tableName = 'Dataframe de Jan 20 09_37_04.xlsx'
fullTablePath = '/'.join([tablePath, tableName])
nnSavesFolder = '../Models/ANNs'
# %% Train and test set
df = pd.read_excel(fullTablePath)
X = mlU.getFeatures(df, 0, -1)
y = mlU.getLabels(df, 'diagnostic')
# %% Split data
X_train, X_test, y_train, y_test = mlU.splitData(X, y, .2)
# %% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
means = sc.mean_
variances = sc.var_
# %% Baseline model
alpha = 8
nFeatures = X.shape[1]
outputNeurons = 1
nSamples = len(X_train)
activations = ['relu', 'relu', 'sigmoid']
l1 = 0.7
l2 = 0.7
dropout = 0.5
batchNorm = False
model = nnU.createANN(alpha=alpha, features=nFeatures, outputNeurons=outputNeurons, nSamples=nSamples,
                      activations=activations, l1=l1, l2=l2, dropout=dropout, batchNorm=batchNorm)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])
modelHistory = model.fit(X_train, y_train, batch_size=500,
                         epochs=800, verbose=2, validation_data=(X_test, y_test))
nnU.plot_history([('Base model', modelHistory)])
yPred = nnU.performance(model, X_test, y_test)
# %%
todayDatetime = todaysDate = datetime.datetime.now()
dateString = re.sub(r':', '_', todaysDate.ctime())[4:-5]

currentNNFolder = f'ANN_{yPred[0].round(2)} date {dateString}'
nnFilename = currentNNFolder+'.pkl'
currentNNPath = '/'.join([nnSavesFolder, currentNNFolder])
qrQuery.makeFolders(currentNNPath)
nnInfoFileName = 'nnInfo.txt'
nnInfoFilePath = '/'.join([currentNNPath, nnInfoFileName])
joinedMeans = ', '.join([str(mean) for mean in means])
joinedVars = ', '.join([str(var) for var in variances])
joinedFeatures = ', '.join(X.columns)
with open(nnInfoFilePath, 'w') as infoFile:
    infoFile.write(f'Means: {joinedMeans}\nVariances: {joinedVars}\nFeatures used: {joinedFeatures}')
mP.saveModel(currentNNPath, nnFilename, model)
