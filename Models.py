"""Conv net model to be used to trin"""
import os
import time
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weight_variable(shape):
    print(shape, "weight variable shape")
    weight = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return weight


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, s=1):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


class Model1:

    def __init__(self, features=[12, 18, 32], kernel_size=[8, 5, 3]):

        self.optimizer = None


        self.weights = []
        self.biases = []

        self.weights.append(weight_variable([kernel_size[0], kernel_size[0], 1, features[0]]))

        self.weights.append(weight_variable([kernel_size[1], kernel_size[1], features[0], features[1]]))

        self.inputs = tf.placeholder(tf.float32, [1, 185, 120, 1])

        self.conv_network = self.conv_architecture(self.inputs)
        self.fc_network = self.fc_architecture(self.conv_network)

        self.output = self.fc_network

        self.expected = tf.placeholder(tf.float32, shape=[4])

        #self.loss = self.mean_squared_loss(output, self.expected)

    def forward_pass(self, session, image):
        # if self.session is None:
        #     self.session = tf.Session()
        #     self.session.run(tf.global_variables_initializer())

        output = session.run(self.output, {self.inputs: image})
        return output


    def train(self, session, training_data, labels, target, epochs=4, learning_rate=1e-3):

        self.loss = self.mean_squared_loss(target, self.output)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        training_function = self.optimizer.minimize(self.loss)

        session.run(tf.global_variables_initializer())

        training_data = np.reshape(training_data, [1, training_data.shape[0], training_data.shape[1], 1])
        for i in range(epochs):

            session.run(training_function, {self.inputs: training_data, self.expected: labels})

        #print(self.session.run(self.loss, {self.inputs: training_data, self.expected: labels}))


    def vectorize(self, x):
        print(x.shape, "conv shape")
        return tf.reshape(x, [1, -1])

    def conv_architecture(self, inputs, append_weights=True):

        """if append_weights is True:
            max_depth = max(len(self.weights), len(self.biases), len(self.strides))
        else:
            max_depth = max(len(self.weights), len(self.biases), len(self.strides))-2"""

        print(inputs.shape, "inputs to conv layer shape")

        conv1 = tf.nn.relu(conv2d(inputs, self.weights[0]))
        print(conv1.shape)
        conv2 = tf.nn.relu(conv2d(conv1, self.weights[1]))
        print(conv2.shape)
        # conv3 = tf.nn.relu(conv2d(conv2, self.weights[2], self.biases[2]))

        return self.vectorize(conv2)

    def fc_architecture(self, vector, num_hidden_neurons=200, append_weights=True):

        if append_weights is True:
            print(vector.shape, "Flatten vector shape")
            self.weights.append(weight_variable([int(vector.shape[1]), num_hidden_neurons]))
            self.biases.append(bias_variable([num_hidden_neurons]))

            self.weights.append(weight_variable([num_hidden_neurons, 4]))
            self.biases.append(bias_variable([4]))

        vector = tf.nn.relu(tf.matmul(vector, self.weights[-2]) + self.biases[-2])
        prediction = tf.nn.softmax(tf.matmul(vector, self.weights[-1]) + self.biases[-1])
        print(prediction.shape)
        return prediction

    def save_variables(self, session, path=None):
        if (path is None):
            path = "./Model1_Variables  {}.ckpt".format(time.strftime("%Y-%m-%d  %H.%M.%S"))
        saver = tf.train.Saver()
        return saver.save(session, path)


    def load_variables(self, path="./Model1_Variables.ckpt"):
        with tf.Session() as session:
            tf.train.Saver().restore(session, path)
            return session

    def mean_squared_loss(self, output, expected):
        return tf.reduce_sum(tf.square(output - expected))
