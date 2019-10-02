import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn_porter import Porter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split, RandomizedSearchCV)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
print(os.getcwd())
# %%
dfForTraining = pd.read_csv('../Dataframes/MoreFeatures.csv').iloc[:, 1:]
# %% Train and test set
y = dfForTraining.loc[:, ['diagnostic']].values
dfForTraining.pop('diagnostic')
X = dfForTraining.iloc[:, :].values
# %% Encode the marker name
le = LabelEncoder()
X[:, -5] = le.fit_transform(X[:, -5])
oh = OneHotEncoder(categorical_features=[-5])
X = oh.fit_transform(X).toarray()
# %% Binarize 'P' and 'N' to 1 and 0
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()
# %% Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
# %%
classifier = RandomForestClassifier(
    n_estimators=200, criterion='entropy', min_samples_split=.0002575)
classifier.fit(X_train, y_train)
# %% confusion matrix preds
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
true_positives = cm[0, 0]
false_positives = cm[1, 0]
true_negatives = cm[1, 1]
false_negatives = cm[0, 1]
accuracy = (true_negatives + true_positives)/(true_positives + \
    false_positives+true_negatives+false_negatives)
print(f'''Accuracy  (Ratio of correct and total preds): {accuracy}''')
print(classification_report(y_test, y_pred,
                            target_names=['Positive', 'Negative']))
#%%
filename = '90_rf.joblib'
joblib.dump(classifier, filename)
#%%
porter = Porter(classifier, language='java')
output = porter.export(embed_data=True)
print(output)