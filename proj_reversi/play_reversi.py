from board_reversi import Board
import tensorflow as tf

weights = {
	'wc1': tf.Variable(tf.random_normal([3, 3, 4, 32])),
	'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
	'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32])),
	'wc4': tf.Variable(tf.random_normal([3, 3, 32, 32])),
	'wd1': tf.Variable(tf.random_normal([8*8*32, 128])),
	'wd2': tf.Variable(tf.random_normal([128, 64]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([32])),
	'bc3': tf.Variable(tf.random_normal([32])),
	'bc4': tf.Variable(tf.random_normal([32])),
	'bd1': tf.Variable(tf.random_normal([128])),
	'bd2': tf.Variable(tf.random_normal([64]))
}

def dense(x, W, b):
	return tf.add(tf.matmul(x, W), b)

def conv2D(x, W, b, strides = 1):
	x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def model(x, weights, biases):
	x = tf.reshape(x, [-1, 8, 8, 4])
	x = conv2D(x, weights['wc1'], biases['bc1'])
	x = conv2D(x, weights['wc2'], biases['bc2'])
	x = conv2D(x, weights['wc3'], biases['bc3'])
	x = conv2D(x, weights['wc4'], biases['bc4'])
	x = tf.reshape(x, [-1, 8*8*32])
	x = dense(x, weights['wd1'], biases['bd1'])
	x = dense(x, weights['wd2'], biases['bd2'])
	return x

board = Board(num_hole = 8, onGUI = True)
s = input()


