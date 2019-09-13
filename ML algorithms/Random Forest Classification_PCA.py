import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
# %%
dfForTraining = pd.read_csv(
    '../Dataframes/PCA_Reduced_Features.csv').iloc[:, 1:]
# %% Train and test set
y = dfForTraining.loc[:, ['diagnostic']].values
dfForTraining.pop('diagnostic')
X = dfForTraining.iloc[:, :].values
# %% Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
# %%
classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', min_samples_split = .0002575)
classifier.fit(X_train, y_train)
# %% confusion matrix preds
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# %% Kfold validation
accuracies = cross_val_score(
    estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
acc_mean = accuracies.mean()
acc_std = accuracies.std()
print(f'Accuracy of model: {acc_mean}')
print(f'Accuracy of model: {acc_std}')
# %% Grid search
n_estimators = np.linspace( 200, 250, 5)
n_estimators = [int(estimator) for estimator in n_estimators]
min_samples_split = np.linspace(.00001, .001, 5)
# %%
parameters = {
    'n_estimators': n_estimators,
    'min_samples_split': [.0002575],
    'criterion': ['entropy']
}
gridSearch = GridSearchCV(
    estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
gridSearch = gridSearch.fit(X_train, y_train)
# Testing grid search
bestAccuracy = gridSearch.best_score_
bestEstimator = gridSearch.best_estimator_
bestParams = gridSearch.best_params_
print(f'Best accuracy: {bestAccuracy}')
#print(f'Best estimator: {bestEstimator}')
print(f'Best parameters: {bestParams}')
