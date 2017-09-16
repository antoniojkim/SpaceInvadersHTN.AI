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


def architecture(input, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3):

    conv1 = conv2d(input, w_conv1, b_conv1, s=2)

    conv2 = conv2d(conv1, w_conv2, b_conv2, s=2)

    conv3 = conv2d(conv2, w_conv3, b_conv3, s=2)

    return conv3


features = [32, 16, 8]
kernel_size = [7, 5, 3]
weights = []
biases = []

for i in range(len(features)):

    weights.append(weight_variable([kernel_size[i],
                                    kernel_size[i],
                                    1,
                                    features[i]]))

    biases.append(bias_variable([features[i]]))

