import numpy as np

def acc(cm):
    """Compute accuracy from Confusion Matrix"""
    
    sm = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                sm += cm[i,j]
                
    return sm / np.sum(cm)