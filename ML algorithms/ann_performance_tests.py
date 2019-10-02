import mlUtils as ml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# %% Importing the dataset
dfForTraining = pd.read_csv(
    '../Dataframes/MoreFeatures.csv').iloc[:, 1:]
y = dfForTraining.loc[:, ['diagnostic']].values
dfForTraining.pop('diagnostic')
X = dfForTraining.values
# %% Binarize 'P' and 'N' to 1 and 0
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()
# %% Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
# %% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# %% Baseline model
alpha = .4
features = X.shape[1]
outputNeurons = 1
nSamples = len(X_train)
activations = ['relu','relu','sigmoid']
l1 = 0.001
dropout = 0.1
batchNorm = False
model = ml.createANN(alpha=alpha, features=features, outputNeurons=outputNeurons, nSamples=nSamples, activations=activations, l1=l1, dropout=dropout, batchNorm=batchNorm)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
modelHistory = model.fit(X_train, y_train, batch_size=2000, epochs=600, verbose=2, validation_data=(X_test, y_test))
#%%
ml.plot_history([('Base model', modelHistory)])
ml.performance(model, X_test, y_test)