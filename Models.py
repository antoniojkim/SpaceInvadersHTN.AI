import random
import time

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(tf.float32, shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, b, s=1):
    return tf.nn.conv2d(x, W, strides=[s, s, s, s], padding='SAME') + b


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class Model1(object):

    def __init__(self, path=None, loss_function=mean_squared_loss,
                 features=[8, 16, 32], kernel_size=[8, 5, 3], strides=[2, 2, 2]):

        self.session = None

        if path != None:
            self.load_variables()
            variables_loaded = True
        else:

            self.strides = strides
            self.weights = []
            self.biases = []


            for kernel, feature in zip(kernel_size, features):
                self.weights.append(weight_variable([kernel, kernel, 1, feature]))

                self.biases.append(bias_variable([feature]))

            variables_loaded = False

        self.input = tf.placeholder(tf.float32)

        self.network = self.conv_architecture(input, strides, variables_loaded)
        self.network = self.fc_architecture(self.network, variables_loaded)

        output = self.network

        self.expected = tf.placeholder(tf.float32)

        self.loss = loss_function(output, self.expected)

    def forward_pass(self, image):
        if self.session == None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        output = self.session.run(self.network, {self.input: image})
        return output

    def train(self, training_data, labels, epochs=10, learning_rate=0.01):
        if self.session == None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        for i in range(epochs):
            r = random.randint(0, len(training_data)-1)

            self.session.run(self.train, {self.input: training_data[r], self.expected: labels[r]})

    def vectorize(self, x):
        return tf.reshape(x, [-1])

    def conv_architecture(self, input, strides, append_weights=True):

        if append_weights == True:
            max_depth = max(len(self.weights), len(self.biases), len(self.strides))-1
        else:
            max_depth = max(len(self.weights), len(self.biases), len(self.strides))-3

        convolution = tf.nn.relu(conv2d(input, self.weights[0], self.biases[0], s=strides[0]))
        for i in range(max_depth):
            convolution = tf.nn.relu(conv2d(convolution, self.weights[i+1], self.biases[i+1], s=strides[i+1]))

        return self.vectorize(convolution)

    def fc_architecture(self, vector, num_hidden_neurons=200, append_weights=True):

        if append_weights == True:
            self.weights.append(weight_variable([vector.shape[1], num_hidden_neurons]))
            self.biases.append(bias_variable([num_hidden_neurons]))

            self.weights.append(weight_variable([num_hidden_neurons, 4]))
            self.biases.append(bias_variable([4]))

        vector = tf.nn.relu(tf.matmul(vector, self.weights[-2]) + self.biases[-2])
        prediction = tf.nn.softmax(tf.matmul(vector, self.weights[-1]) + self.biases[-1])

        return prediction

    def save_variables(self, path=None):
        if (path == None):
            path = "./Model1_Variables  {}.ckpt".format(time.strftime("%Y-%m-%d  %H.%M.%S"))
        if self.session == None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        return saver.save(self.session, path)


    def load_variables(self, path="./Model1_Variables.ckpt"):
        if self.session == None:
            self.session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.session, path)

def mean_squared_loss(self, output, expected):
    return tf.reduce_sum(tf.square(output - expected))