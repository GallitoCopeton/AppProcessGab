import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from sklearn.ensemble import GradientBoostingClassifier
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
classifier = GradientBoostingClassifier(loss='deviance', learning_rate=.09, n_estimators=250, subsample=.9)
classifier.fit(X_train, y_train)
report = mP.getReport(classifier, X_train, y_train, X_test, y_test)