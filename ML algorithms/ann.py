# Artificial Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Data Preprocessing

import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, OneHotEncoder,
                                   StandardScaler)

# %% Importing the dataset
dfForTraining = pd.read_csv('Dataframes/test.csv').iloc[:, 1:]
y = dfForTraining.loc[:, ['diagnostic']].values
dfForTraining.pop('diagnostic')
# dfForTraining.pop('marker')

X = dfForTraining.iloc[:, :].values
# %% Encode the marker name
le = LabelEncoder()
X[:, -5] = le.fit_transform(X[:, -5])
oh = OneHotEncoder(categorical_features=[-5])
X = oh.fit_transform(X).toarray()
# %% Binarize 'P' and 'N' to 1 and 0
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()
# %% Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
# %% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# %% Part 2 - Now let's make the ANN!

# %% Importing the Keras libraries and packages

# %% Baseline model
alpha = 2
features = X.shape[1]
outputNeurons = 1
Nh = int((len(X_train))/(alpha*(features+outputNeurons)))
baseline_model = Sequential([
    Dense(units=Nh, kernel_initializer='uniform',
          activation='relu', input_dim=features),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
])

# Compiling the ANN
baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
                       'accuracy', 'binary_crossentropy'])
baseline_history = baseline_model.fit(
    X_train, y_train, batch_size=510, epochs=500, verbose=2, validation_data=(X_test, y_test))
# %% Smaller model
Nh = int(Nh/4)
smaller_model = Sequential([
    Dense(units=Nh, kernel_initializer='uniform',
          activation='relu', input_dim=features),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
])

# Compiling the ANN
smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
    'accuracy', 'binary_crossentropy'])
smaller_history = smaller_model.fit(
    X_train, y_train, batch_size=510, epochs=500, verbose=2, validation_data=(X_test, y_test))
# %% Bigger model
Nh = int(Nh*4)
bigger_model = Sequential([
    Dense(units=Nh, kernel_initializer='uniform',
          activation='relu', input_dim=features),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
])

# Compiling the ANN
bigger_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
    'accuracy', 'binary_crossentropy'])
bigger_history = bigger_model.fit(
    X_train, y_train, batch_size=510, epochs=500, verbose=2, validation_data=(X_test, y_test))
# %% Baseline model with dropout
alpha = 2
features = X.shape[1]
outputNeurons = 1
Nh = int((len(X_train))/(alpha*(features+outputNeurons)))
baseline_model_dropout = Sequential([
    Dense(units=Nh, kernel_initializer='uniform',
          activation='relu', input_dim=features),
    Dropout(0.2),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.001),
          bias_regularizer=keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
])

# Compiling the ANN
baseline_model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
    'accuracy', 'binary_crossentropy'])
baseline_history_dropout = baseline_model_dropout.fit(
    X_train, y_train, batch_size=510, epochs=500, verbose=2, validation_data=(X_test, y_test))
# %% Baseline model with dropout and BN
alpha = 2
features = X.shape[1]
outputNeurons = 1
Nh = int((len(X_train))/(alpha*(features+outputNeurons)))
baseline_model_dropout_bn = Sequential([
    Dense(units=Nh, kernel_initializer='uniform',
          activation='relu', input_dim=features),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
])

# Compiling the ANN
baseline_model_dropout_bn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
    'accuracy', 'binary_crossentropy'])
baseline_history_dropout_bn = baseline_model_dropout_bn.fit(
    X_train, y_train, batch_size=1000, epochs=200, verbose=2, validation_data=(X_test, y_test))
# %% Baseline model with dropout and BN more hl
alpha = 2
features = X.shape[1]
outputNeurons = 1
Nh = int((len(X_train))/(alpha*(features+outputNeurons)))
baseline_model_dropout_bn = Sequential([
    Dense(units=Nh, kernel_initializer='uniform',
          activation='relu', input_dim=features),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=Nh,
          kernel_initializer='uniform',
          activation='relu',
          use_bias=True,
          kernel_regularizer=keras.regularizers.l2(0.0001),
          bias_regularizer=keras.regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
])

# Compiling the ANN
baseline_model_dropout_bn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
    'accuracy', 'binary_crossentropy'])
baseline_history_dropout_bn = baseline_model_dropout_bn.fit(
    X_train, y_train, batch_size=1000, epochs=200, verbose=2, validation_data=(X_test, y_test))
# %% Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = baseline_model.predict(X_test)
y_predc = (y_pred > 0.5)
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

# %%


def plot_history(histories, key='binary_crossentropy'):
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


plot_history([('baseline_dropout', baseline_history_dropout),
              ('baseline_dropout_bn', baseline_history_dropout_bn)], key='binary_crossentropy')
