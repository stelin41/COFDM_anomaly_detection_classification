import numpy as np
from collections import Counter

def predict_labels(y_pred: np.array, N:int, array_length: int) -> np.array:
    predicted_labels = np.zeros(N, dtype=np.int8)

    for i in range(N):
        group_labels = y_pred[i*array_length:(i+1)*array_length] 
        label_count = Counter(group_labels)
        if 1 in label_count or 2 in label_count:
            most_common_label = label_count.most_common(1)[0][0]
            predicted_labels[i] = most_common_label
        else:
            predicted_labels[i] = 0
            
    return predicted_labels
