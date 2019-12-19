from sklearn.model_selection import GridSearchCV

import machineLearningUtilities.dataPreparation as mlU
from machineLearningUtilities import modelPerformance as mP
from sklearn.ensemble import RandomForestClassifier
# %%
dfForTraining = mlU.loadData('../Dataframe creation/csvs/lessF.csv')
# %% Train and test set
X = mlU.getFeatures(dfForTraining, 0, -1)
y = mlU.getLabels(dfForTraining, 'diagnostic')
# %% Split data
X_train, X_test, y_train, y_test = mlU.splitData(X, y)
# %% Feature Scaling
X_train = mlU.scaleData(X_train)
X_test = mlU.scaleData(X_test)
# %%
classifier = RandomForestClassifier(n_estimators=800, min_samples_split=.0004, max_depth=9000, criterion='entropy')
classifier.fit(X_train, y_train)
report = mP.getReport(classifier, X_train, y_train, X_test, y_test)
#%%
filePath = './models'
fileName = 'RandomForestClassifier.pkl'

mP.saveModel(filePath, fileName, classifier)
#%% Set the parameters by cross-validation
tuned_parameters = [{'criterion': ['gini'], 'n_estimators': [1000, 1200], 'min_samples_split': [.0002, .0003]},
                    {'criterion': ['entropy'], 'n_estimators': [800, 1000], 'min_samples_split': [.0002, .0004]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
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