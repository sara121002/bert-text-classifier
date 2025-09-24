from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def print_report(labels, preds, target_names=None):
    print(classification_report(labels, preds, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))
