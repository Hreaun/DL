def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    correct = sum(a == b for a, b in zip(prediction, ground_truth))

    true_pos = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(prediction)):
        if prediction[i]:
            true_pos += (prediction[i] == ground_truth[i])
            false_pos += (prediction[i] != ground_truth[i])
        else:
            false_neg += (prediction[i] != ground_truth[i])

    accuracy = correct / len(ground_truth)

    if true_pos + false_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos + false_neg == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    correct = sum(a == b for a, b in zip(prediction, ground_truth))

    accuracy = correct / len(ground_truth)

    return accuracy
