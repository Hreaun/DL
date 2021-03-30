import numpy as np

from assignment3.layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # Create necessary layers
        image_width, image_height, n_channels = input_shape
        padding = 1
        pool_size = 4
        filter_size = 3
        stride = 1

        def conv_out_volume(w_in, h_in, F, padding_size, S, num_of_filters):
            w_out = (w_in - F + 2 * padding_size) / S + 1
            h_out = (h_in - F + 2 * padding_size) / S + 1
            return int(w_out), int(h_out), num_of_filters

        def max_pool_out_volume(w_in, h_in, F, S, num_of_filters):
            w_out = (w_in - F) / S + 1
            h_out = (h_in - F) / S + 1
            return int(w_out), int(h_out), num_of_filters

        conv1_out_volume = conv_out_volume(image_width, image_height,
                                           filter_size, padding,
                                           stride, conv1_channels)

        max_pool1_out_volume = max_pool_out_volume(conv1_out_volume[0],
                                                   conv1_out_volume[1],
                                                   pool_size,
                                                   pool_size,
                                                   conv1_out_volume[2])

        conv2_out_volume = conv_out_volume(max_pool1_out_volume[0],
                                           max_pool1_out_volume[1],
                                           filter_size, padding,
                                           stride, conv2_channels)

        max_pool2_out_volume = max_pool_out_volume(conv2_out_volume[0],
                                                   conv2_out_volume[1],
                                                   pool_size,
                                                   pool_size,
                                                   conv2_out_volume[2])

        self.conv1 = ConvolutionalLayer(n_channels, conv1_channels, filter_size, padding)
        self.relu1 = ReLULayer()
        self.max_pool1 = MaxPoolingLayer(pool_size, pool_size)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding)
        self.relu2 = ReLULayer()
        self.max_pool2 = MaxPoolingLayer(pool_size, pool_size)
        self.flatten = Flattener()
        self.fully_connected = FullyConnectedLayer(max_pool2_out_volume[0] *
                                                   max_pool2_out_volume[1] *
                                                   max_pool2_out_volume[2],
                                                   n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for v in self.params().values():
            v.grad = np.zeros(v.grad.shape)

        # Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        out = [None]
        out[0] = self.conv1.forward(X)
        out.append(self.relu1.forward(out[0]))
        out.append(self.max_pool1.forward(out[1]))
        out.append(self.conv2.forward(out[2]))
        out.append(self.relu2.forward(out[3]))
        out.append(self.max_pool2.forward(out[4]))
        out.append(self.flatten.forward(out[5]))
        out.append(self.fully_connected.forward(out[6]))

        d_out = [None]
        loss, d_out[0] = softmax_with_cross_entropy(out[7], y)
        d_out.append(self.fully_connected.backward(d_out[0]))
        d_out.append(self.flatten.backward(d_out[1]))
        d_out.append(self.max_pool2.backward(d_out[2]))
        d_out.append(self.relu2.backward(d_out[3]))
        d_out.append(self.conv2.backward(d_out[4]))
        d_out.append(self.max_pool1.backward(d_out[5]))
        d_out.append(self.relu1.backward(d_out[6]))
        d_out.append(self.conv1.backward(d_out[7]))

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        conv1_out = self.conv1.forward(X)
        relu1_out = self.relu1.forward(conv1_out)
        max_pool1_out = self.max_pool1.forward(relu1_out)
        conv2_out = self.conv2.forward(max_pool1_out)
        relu2_out = self.relu2.forward(conv2_out)
        max_pool2_out = self.max_pool2.forward(relu2_out)
        flatten_out = self.flatten.forward(max_pool2_out)
        fully_connected_out = self.fully_connected.forward(flatten_out)
        pred = np.argmax(fully_connected_out, axis=1)

        return pred

    def params(self):
        result = {
            'conv1_W': self.conv1.W,
            'conv1_B': self.conv1.B,
            'conv2_W': self.conv2.W,
            'conv2_B': self.conv2.B,
            'fully_connected_W': self.fully_connected.W,
            'fully_connected_B': self.fully_connected.B
        }

        # Aggregate all the params from all the layers
        # which have parameters

        return result
