import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W: np array - weights
      reg_strength: - float value

    Returns:
      loss: single value - l2 regularization loss
      gradient: np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions: np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs: np array of the same shape as predictions -
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
      probs: np array, shape is either (N) or (batch_size, N) -
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
      predictions: np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
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


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        # forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.cache = X
        return np.maximum(0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = np.multiply(d_out, np.int64(self.cache > 0))
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
                Backward pass
                Computes gradient with respect to input and
                accumulates gradients within self.W and self.B

                Arguments:
                d_out, np array (batch_size, n_output) - gradient
                   of loss function with respect to output

                Returns:
                d_result: np array (batch_size, n_input) - gradient
                  with respect to input
                """
        # backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad += self.X.T @ d_out

        # производная dL/dB = d_out(dL/dZ) * вектор из 1 размерности B
        self.B.grad += np.sum(d_out, axis=0, keepdims=True)

        d_input = d_out @ self.W.value.T
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        """
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        """

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape

        stride = 1
        out_height = int((height - self.filter_size + 2 * self.padding) / stride + 1)
        out_width = int((width - self.filter_size + 2 * self.padding) / stride + 1)

        # Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                # Implement forward pass for specific location
                out[:, y, x, :] = np.sum(
                    X[:, y:self.filter_size + y, x:self.filter_size + x, :, np.newaxis]
                    * self.W.value[np.newaxis, :],
                    axis=(1, 2, 3)
                )

        return out + self.B.value

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                pass

        raise Exception("Not implemented!")

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        """
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        """
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
