import numpy as np

from assignment2.layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        # Create necessary layers
        self.reg = reg
        self.hidden_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.fully_connected_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for v in self.params().values():
            v.grad = np.zeros(v.grad.shape)

        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        out = [None]
        out[0] = self.hidden_layer.forward(X)
        out.append(self.relu_layer.forward(out[0]))
        out.append(self.fully_connected_layer.forward(out[1]))
        d_out = [None]
        loss, d_out[0] = softmax_with_cross_entropy(out[2], y)
        d_out.append(self.fully_connected_layer.backward(d_out[0]))
        d_out.append(self.relu_layer.backward(d_out[1]))
        d_out.append(self.hidden_layer.backward(d_out[2]))

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        for v in self.params().values():
            l2_loss, l2_grad = l2_regularization(v.value, self.reg)
            loss += l2_loss
            v.grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        # pred = np.zeros(X.shape[0], np.int)

        hl_out = self.hidden_layer.forward(X)
        relu_out = self.relu_layer.forward(hl_out)
        pred = np.argmax(self.fully_connected_layer.forward(relu_out), axis=1)

        return pred

    def params(self):
        # Implement aggregating all of the params
        result = {'hidden_W': self.hidden_layer.W,
                  'fully_connected_W': self.fully_connected_layer.W,
                  'hidden_B': self.hidden_layer.B,
                  'fully_connected_B': self.fully_connected_layer.B}
        return result
