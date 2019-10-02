import logging

import keras
import pandas as pd
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

logger_ann = logging.getLogger(__name__)
logger_ann.setLevel(logging.DEBUG)
formatter_ann = logging.Formatter(
    '%(levelname)s:%(name)s:%(message)s')
fileHandler_ann = logging.FileHandler('ann.log')
fileHandler_ann.setFormatter(formatter_ann)
logger_ann.addHandler(fileHandler_ann)


def createANN(alpha: float, features: int, outputNeurons: int, nSamples: int, activations: list, l2: float = None, l1: float = None, dropout: float = None, batchNorm: bool = False):
    '''Creates a neural network which has the defined parameters
    Returns: annModel'''
    hiddenUnits = int((nSamples)/(alpha*(features+outputNeurons)))
    model = Sequential()
    logger_ann.info(f'''Inputs:
    alpha: {alpha}
    features: {features}
    outputNeurons: {outputNeurons}
    nSamples: {nSamples}
    activations: {activations}
    l2: {l2}
    l1: {l1}
    dropout: {dropout}
    batchNorm: {batchNorm}''')
    logger_ann.info('Starting loop...')

    for i, activation in enumerate(activations):
        if i == 0:
            model.add(
                Dense(units=hiddenUnits, kernel_initializer='uniform',
                      activation=activation, input_dim=features)
            )
            if batchNorm:
                model.add(BatchNormalization())
            if dropout:
                model.add(Dropout(dropout))
            logger_ann.debug(f'''Input layer done.''')
            continue
        if i == len(activations)-1:
            model.add(
                Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
            )
            logger_ann.debug(f'''Output layer done.''')
            continue
        if l2:
            model.add(
                Dense(units=hiddenUnits,
                      kernel_initializer='uniform',
                      activation=activation,
                      use_bias=True,
                      kernel_regularizer=keras.regularizers.l2(l2),
                      bias_regularizer=keras.regularizers.l2(l2))
            )
        elif l1:
            model.add(
                Dense(units=hiddenUnits,
                      kernel_initializer='uniform',
                      activation=activation,
                      use_bias=True,
                      kernel_regularizer=keras.regularizers.l1(l1),
                      bias_regularizer=keras.regularizers.l1(l1))
            )
        elif l1 and l2:
            model.add(
                Dense(units=hiddenUnits,
                      kernel_initializer='uniform',
                      activation=activation,
                      use_bias=True,
                      kernel_regularizer=keras.regularizers.l1_l2(
                          l1=l1, l2=l2),
                      bias_regularizer=keras.regularizers.l1_l2(
                          l1=l1, l2=l2))
            )
        else:
            model.add(
                Dense(units=hiddenUnits,
                      kernel_initializer='uniform',
                      activation=activation
                      )
            )
        if batchNorm:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout))
        logger_ann.debug(f'Hidden layer no. {i} done.')
    return model


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


def performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
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
    accuracy = (true_negatives + true_positives)/len(y_test)
    print(
        f'''Sensitivity (How good am I at detecting true positives): {sensitivity}''')
    print(
        f'''Specificity  (How good am I at detecting true negatives): {specificity}''')
    print(f'''Accuracy  (Ratio of correct and total preds): {accuracy}''')
