import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def perform_lda(energy_dif_matrix: np.array, labels:np.array):
    model_da = LinearDiscriminantAnalysis().fit(energy_dif_matrix,labels)
    return model_da

def perform_qda(energy_dif_matrix: np.array, labels:np.array):
    model_da = QuadraticDiscriminantAnalysis().fit(energy_dif_matrix,labels)
    return model_da
