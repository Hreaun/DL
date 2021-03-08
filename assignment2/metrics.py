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
