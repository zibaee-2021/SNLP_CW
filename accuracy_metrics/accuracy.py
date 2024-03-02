"""
Compute accuracy.

For Datasets.py-QA task b, yes/no questions,
calculate simple percentage of questions for which the system provides the correct answer.


This function simply compares the true labels to the predicted labels, counts the number of correct predictions, and divides by the total number of predictions to get the accuracy.

The measurement in example snippet below is most useful when the classes are balanced.
If your classes are imbalanced (significantly more 'yes' than 'no' answers or vice versa),
other metrics might be necessary as well, such as precision, recall, F1-score.
"""
import numpy as np


def accuracy(y_true, y_pred):
    """
    Computes the accuracy, which is the proportion of correct predictions over total predictions.

    Parameters:
    - y_true: np.array, true binary labels (0 or 1).
    - y_pred: np.array, predicted binary labels (0 or 1).

    Returns:
    - accuracy: float, the accuracy of the predictions.
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)

    accuracy = correct_predictions / total_predictions
    return accuracy


# Example usage:
y_true = np.array([1, 0, 1, 1, 0]) # true labels
y_pred = np.array([1, 1, 1, 0, 0]) # predicted labels

acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
