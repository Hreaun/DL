import numpy as np


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    if predictions.ndim == 1:
        normalized_predictions = predictions - np.max(predictions)
        probs = np.exp(normalized_predictions) / np.sum(np.exp(normalized_predictions))
    else:
        normalized_predictions = predictions - np.max(predictions, axis=1)[:, None]
        probs = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, normalized_predictions)
    return probs


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        target_index_vect = target_index.reshape(-1, 1)
        loss = -np.sum(np.log(np.take_along_axis(probs, target_index_vect, axis=1))) / probs.shape[0]

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of loss by predictions
    """
    y_true = np.zeros(predictions.shape)
    if predictions.ndim == 1:
        y_true[target_index] = 1
    else:
        target_index_vect = target_index.reshape(-1, 1)
        np.put_along_axis(y_true, target_index_vect, 1, axis=1)

    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprediction = (probs - y_true)
    if predictions.ndim != 1:
        dprediction /= predictions.shape[0]

    return loss, dprediction


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of l2 loss by weight
    """

    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad


def linear_softmax(X, W, target_index):
    """
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of loss by weight
    """

    # получаем вероятности нашей модели (y = x * w)
    predictions = np.dot(X, W)

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)

    # loss func = -(w * x)_target + ln( sum(e^(w*x)_target) )
    # => dL/dW = X * (dL/dpred)
    dL = X.transpose() @ dprediction

    return loss, dL


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        """
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        """

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            loss = 0
            for i in range(num_train // batch_size):
                batch = X[batches_indices[i]]
                loss, gradient = linear_softmax(batch, self.W, y[batches_indices[i]])
                l2_loss, l2_grad = l2_regularization(self.W, reg)
                loss += l2_loss
                gradient += l2_grad
                self.W -= learning_rate * gradient
                loss_history.append(loss)
                # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        y_pred = np.argmax(X @ self.W, axis=1)

        return y_pred
