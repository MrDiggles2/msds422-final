import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

def display_metrics(model_name, y_test, y_pred, y_scores, pos_label = None):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)

    print(f"{model_name}")
    print(f"\tAccuracy: {accuracy}")
    print(f"\tPrecision: {precision}")
    print(f"\tRecall: {recall}")
    print(f"\tTPR: {tpr[1]}")
    print(f"\tFPR: {fpr[1]}")

    # ROC Curve

    fpr, tpr, _ = metrics.roc_curve(y_test, y_scores, pos_label=pos_label)
    roc_auc = metrics.roc_auc_score(y_test, y_scores)

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc='lower right')
    
    # Precision-Recall Curve

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_scores, pos_label=pos_label)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')

    plt.show()