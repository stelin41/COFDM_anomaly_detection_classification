import numpy as np
from sklearn import svm

# Tiene parámetro class_weight, se puede tratar con clases desbalanceadas
def svc(energy_dif_matrix: np.array, sample_labels: np.array) -> svm.SVC:
    clf = svm.SVC().fit(X=energy_dif_matrix, y=sample_labels)
    return clf

def linear_svc(energy_dif_matrix: np.array, sample_labels: np.array) -> svm.LinearSVC:
    clf = svm.LinearSVC().fit(X=energy_dif_matrix, y=sample_labels)
    return clf

def nu_svc(energy_dif_matrix: np.array, sample_labels: np.array) -> svm.NuSVC:
    clf = svm.NuSVC().fit(X=energy_dif_matrix, y=sample_labels)
    return clf