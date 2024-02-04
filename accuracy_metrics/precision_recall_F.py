"""
Compute precision, recall and F1-score.

For BioASQ-QA task b, for list questions,
these metrics are used to evaluate the correctness and completeness of the lists provided by the systems.


The example snippets below use either numpy to manually implement prF1 or scikit-learn.
For more control over how the metrics are computed or if there is a need to tweak the calculations for
specific cases, the manual numpy approach might be preferable.
Otherwise the built-in functions from scikit-learn are sufficient.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def precision(y_true, y_pred):
    true_positives = np.sum(y_pred[y_true == 1])
    predicted_positives = np.sum(y_pred)
    precision = true_positives / predicted_positives if predicted_positives else 0
    return precision


def recall(y_true, y_pred):
    true_positives = np.sum(y_pred[y_true == 1])
    actual_positives = np.sum(y_true)
    recall = true_positives / actual_positives if actual_positives else 0
    return recall


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0


# Example usage:
y_true = [0, 1, 1, 0, 1] # true labels
y_pred = [0, 1, 0, 1, 1] # predicted labels

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")


y_true = np.array([1, 0, 1, 1, 0]) # true labels
y_pred = np.array([1, 1, 1, 0, 0]) # predicted labels

# If using scikit-learn
precision_val = precision_score(y_true, y_pred)
recall_val = recall_score(y_true, y_pred)
f1_val = f1_score(y_true, y_pred)

print(f"Precision: {precision_val}")
print(f"Recall: {recall_val}")
print(f"F1-score: {f1_val}")