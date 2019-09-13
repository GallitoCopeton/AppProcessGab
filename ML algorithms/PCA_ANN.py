# -*- coding: utf-8 -*-

# Artificial Neural Network with PCA feature preprocessing


import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import (
    LabelBinarizer, LabelEncoder, OneHotEncoder, StandardScaler)


# %% PCA
# Prepare data, keep diagnostic column before deleting it
originalDF = pd.read_csv('../Dataframes/markersDF.csv').iloc[:, 1:]
y = originalDF.loc[:, ['diagnostic']]
del originalDF['diagnostic']
labelBinarizer = LabelBinarizer()
originalDF = originalDF.drop('marker', axis=1)
y = labelBinarizer.fit_transform(y)
scaler = StandardScaler()
scaler.fit(originalDF.iloc[:, :])
scaledDF = scaler.transform(originalDF.iloc[:, :])
# Perform pca on a for loop for different components amount
n_components = 8
innertias = []
components = []
for n in range(2, n_components+1):
    pca = PCA(n_components=n)
    pca.fit(scaledDF)
    x_pca = pca.transform(scaledDF)
    innertias.append(pca.explained_variance_ratio_.sum())
    components.append(n)
fig, ax = plt.subplots()
ax.plot(components, innertias)
columns = []
for i in range(0, n_components):
    columns.append(f'Component{i+1}')
reducedDF = pd.DataFrame(x_pca, columns=columns)
# %% Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    reducedDF, y, test_size=.20)
# %% Fitting the ANN to the Training set
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=100, kernel_initializer='uniform',
                     activation='relu', input_dim=8))

# Adding the second hidden layer
classifier.add(
    Dense(units=2000, kernel_initializer='uniform', activation='relu'))

# Adding the third hidden layer
classifier.add(
    Dense(units=800, kernel_initializer='uniform', activation='relu'))

# Adding the fourth hidden layer
classifier.add(
    Dense(units=120, kernel_initializer='uniform', activation='relu'))

# Adding the fifth hidden layer
classifier.add(Dense(units=100, kernel_initializer='uniform',
                     activation='exponential'))

# Adding the output layer
classifier.add(
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
model_history = classifier.fit(
    X_train, y_train, batch_size=100000, epochs=200, validation_data=(X_test, y_test))
# %% Fitting the ANN to the Training set
classifier2 = Sequential()

# Adding the input layer and the first hidden layer
classifier2.add(Dense(units=100, kernel_initializer='uniform',
                     activation='relu', input_dim=8))

# Adding the second hidden layer
classifier2.add(
    Dense(units=2000, kernel_initializer='uniform', activation='relu'))

# Adding the third hidden layer
classifier2.add(
    Dense(units=800, kernel_initializer='uniform', activation='relu'))

# Adding the fourth hidden layer
classifier2.add(
    Dense(units=120, kernel_initializer='uniform', activation='linear'))

# Adding the fifth hidden layer
#classifier2.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'linear'))

# Adding the output layer
classifier2.add(
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier2.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_history2 = classifier2.fit(
    X_train, y_train, batch_size=1000000, epochs=1500, validation_data=(X_test, y_test))

# %%


def plot_history(histories, key='acc'):
    plt.figure(figsize=(8, 5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


plot_history([('l2', model_history2)])
# %% Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier2.predict(X_test)
y_predc = (y_pred > 0.7)
print(classification_report(y_test, y_predc,
                            target_names=['Positive', 'Negative']))
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_predc)
true_positives = cm[0, 0]
false_positives = cm[1, 0]
true_negatives = cm[1, 1]
false_negatives = cm[0, 1]
sensitivity = true_positives/(true_positives+false_negatives)
specificity = true_negatives/(true_negatives+false_positives)
accuracy = (true_negatives + true_positives)/473
print(
    f'''Sensitivity (How good am I at detecting true positives): {sensitivity}''')
print(
    f'''Specificity  (How good am I at detecting true negatives): {specificity}''')
print(f'''Accuracy  (Ratio of correct and total preds): {accuracy}''')
