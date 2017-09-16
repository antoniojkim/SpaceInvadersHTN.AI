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

def forwards_pass(x, w, b):

    return tf.matmul(w, x) + b

def relu(x):

    return tf.nn.relu(x)

def softmax(x):

    return tf.nn.softmax(x)

def tanh(x):

    return tf.nn.tanh(x)

def vectorize(x):

    return tf.reshape(x, [-1])


def conv_architecture(input, weights, biases, strides):

    conv1 = relu(conv2d(input, weights[0], biases[0], s=strides[0]))

    conv2 = relu(conv2d(conv1, weights[1], biases[1], s=strides[1]))

    conv3 = relu(conv2d(conv2, weights[2], biases[2], s=strides[2]))

    return vectorize(conv3)


def fc_architecture(vector, weights, biases):

    weights.append(weight_variable([vector.shape[1], 200]))
    biases.append(bias_variable([200]))

    vector = relu(tf.matmul(vector, weights[-1], biases[-1]))

    weights.append(weight_variable([200, 4]))
    biases.append(bias_variable([4]))

    prediction = softmax(tf.matmul(vector, weights[-1], biases[-1]))

    return prediction

features = [8, 16, 32]
kernel_size = [8, 5, 3]
strides = [2, 2, 2]
weights = []
biases = []

for i in range(len(features)):

    weights.append(weight_variable([kernel_size[i],
                                    kernel_size[i],
                                    1,
                                    features[i]]))

    biases.append(bias_variable([features[i]]))
