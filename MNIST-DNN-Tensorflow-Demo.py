import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None,):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=1.0 / math.sqrt(float(in_size))))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

layer1 = add_layer(xs, 784, 500,  activation_function=tf.nn.relu)
layer2 = add_layer(layer1, 500, 500,  activation_function=tf.nn.relu)
prediction = add_layer(layer2, 500, 10,  activation_function=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(12000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 600 == 0:
        print("Accuracy of Testing Set:{:.3f} in epoch {:0>2d}".format(compute_accuracy(mnist.test.images, mnist.test.labels),int(i/600+1)))