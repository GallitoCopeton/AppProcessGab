import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import (
    LabelBinarizer, LabelEncoder, OneHotEncoder, StandardScaler)
from sklearn.svm import SVC
# %%
dfForTraining = pd.read_csv(
    '../Dataframes/PCA_Reduced_Features.csv').iloc[:, 1:]
# %% Train and test set
y = dfForTraining.loc[:, ['diagnostic']].values
dfForTraining.pop('diagnostic')
X = dfForTraining.iloc[:, :].values
# %% Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
# %% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# %%
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
# %%
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
true_positives = cm[0,0]
false_positives = cm[1,0]
true_negatives = cm[1,1]
false_negatives = cm[0,1]
sensitivity = true_positives/(true_positives+false_negatives)
specificity = true_negatives/(true_negatives+false_positives)
accuracy = (true_negatives + true_positives)/473
print(f'''Sensitivity (How good am I at detecting positives): {sensitivity}''')
print(f'''Specificity  (How good am I at avoiding false alarms): {specificity}''')
print(f'''Accuracy  (Ratio of correct and total preds): {accuracy}''')
# %% Kfold validation
accuracies = cross_val_score(
    estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
acc_mean = accuracies.mean()
acc_std = accuracies.std()
print(f'Accuracy of model mean: {acc_mean}')
print(f'Accuracy of model std: {acc_std}')
# %% Grid search
C = np.linspace(10, 300, 8)
C = [int(Ci) for Ci in C]
gamma = np.linspace(.0001, 1, 3)
parameters = [
    {
        'C': C,
        'kernel': ['rbf'],
        'gamma': gamma
    }]
gridSearch = GridSearchCV(
    estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
gridSearch = gridSearch.fit(X_train, y_train.ravel())
# Testing grid search
bestAccuracy = gridSearch.best_score_
bestEstimator = gridSearch.best_estimator_
bestParams = gridSearch.best_params_
print(f'Best accuracy: {bestAccuracy}')
#print(f'Best estimator: {bestEstimator}')
print(f'Best parameters: {bestParams}')
