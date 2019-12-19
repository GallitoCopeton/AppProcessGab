import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from sklearn.neighbors import KNeighborsClassifier
# %%
dfForTraining = mlU.loadData('../Dataframes/estesi.csv')
# %% Train and test set
X = mlU.getFeatures(dfForTraining, 0, -1)
y = mlU.getLabels(dfForTraining, 'diagnostic')
# %% Split data
X_train, X_test, y_train, y_test = mlU.splitData(X, y)
# %% Feature Scaling
X_train = mlU.scaleData(X_train)
X_test = mlU.scaleData(X_test)
# %%
classifier = KNeighborsClassifier(n_neighbors=30, weights='distance', algorithm='ball_tree', leaf_size=60, metric='minkowski', n_jobs=-1)
classifier.fit(X_train, y_train)
report = mP.getReport(classifier, X_train, y_train, X_test, y_test)