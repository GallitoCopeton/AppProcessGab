import numpy as np
from sklearn.model_selection import GridSearchCV

import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from sklearn.svm import SVC
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
classifier = SVC(kernel='rbf', C=210, gamma=.00001)
classifier.fit(X_train, y_train)
report = mP.getReport(classifier, X_train, y_train, X_test, y_test)
#%%
filePath = '../../models'
fileName = 'K_SVM.pkl'

mP.saveModel(filePath, fileName, clf)
#%%
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    mP.getReport(clf, X_train, y_train, X_test, y_true)
    print()