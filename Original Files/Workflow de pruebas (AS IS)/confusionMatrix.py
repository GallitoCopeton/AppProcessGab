#Funcion para obtener matriz de confusion
import numpy as np

def confusionMatrix(real, predicted):
    matrix = np.zeros(shape=(5))
     # acomodados de la siguietne manera: TP, FP, TN, FN, Indeterminado(por cualquier de los lados)
    pos = "P"
    neg = "N"
    ind = "I"

    for i,test in enumerate(real):
        if (real[i].upper() == pos and predicted[i].upper() == pos):
            matrix[0] = matrix[0] + 1
        if (real[i].upper() == neg and predicted[i].upper() == pos):
             matrix[1] = matrix[1] + 1
        if (real[i].upper() == neg and predicted[i].upper() == neg):
             matrix[2] = matrix[2] + 1
        if (real[i].upper() == pos and predicted[i].upper() == neg):
             matrix[3] = matrix[3] + 1
        if (real[i].upper() == ind or predicted[i].upper() == ind):
             matrix[4] = matrix[4] + 1

    tp = matrix[0]
    fp = matrix[1]
    tn = matrix[2]
    fn = matrix[3]
    
    dictStats={}
    dictStats['Precision'] = (tp/(tp+fp))
    dictStats['Sensitivity'] = (tp/(tp+fn))
    dictStats['Specificity'] = (tn/(tn+fp))
    dictStats['Negative Predictive Value'] = (tn/(tn+fn))
    dictStats['Indeterminados'] = matrix[4]
    dictStats['Total de pruebas'] = len(real)
    
    return dictStats