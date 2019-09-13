import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split, RandomizedSearchCV)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
print(os.getcwd())
# %%
dfForTraining = pd.read_csv('Dataframes/test.csv').iloc[:, 1:]
# %% Train and test set
y = dfForTraining.loc[:, ['diagnostic']].values
dfForTraining.pop('diagnostic')
dfForTraining.pop('marker')
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
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

# %% Random grid search
param_dist = dict(n_estimators=list(range(10, 300)),
                  max_depth=list(range(1, 10)),
                  min_samples_leaf=list(range(1, 10)))
rand = RandomizedSearchCV(classifier, param_dist, cv=10,
                          scoring='neg_mean_squared_error', n_iter=30)
rand.fit(X_train, y_train)
# %%
y_pred_rand = rand.predict(X_test)
cm_rand = confusion_matrix(y_test, y_pred_rand)
true_positives = cm_rand[0, 0]
false_positives = cm_rand[1, 0]
true_negatives = cm_rand[1, 1]
false_negatives = cm_rand[0, 1]
accuracy = (true_negatives + true_positives)/len(y_test)
bestAccuracy = rand.best_score_
bestEstimator = rand.best_estimator_
bestParams = rand.best_params_
mse = mean_squared_error(y_test, y_pred_rand)
print(f'''Accuracy: {accuracy}''')
print(f'''MSE: {mse}''')
print(classification_report(y_test, y_pred_rand,
                            target_names=['Positive', 'Negative']))
# %%
new_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=0.0002575,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
new_classifier.fit(X_train, y_train)
y_pred = new_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
true_positives = cm[0, 0]
false_positives = cm[1, 0]
true_negatives = cm[1, 1]
false_negatives = cm[0, 1]
accuracy = (true_negatives + true_positives)/(true_positives +
                                              false_positives+true_negatives+false_negatives)
print(f'''Accuracy  (Ratio of correct and total preds): {accuracy}''')
print(classification_report(y_test, y_pred,
                            target_names=['Positive', 'Negative']))
# %%
print(len(dfForTraining.columns))
# %% 
# Creating a bar plot
feature_imp = pd.Series(classifier.feature_importances_,index=dfForTraining.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
