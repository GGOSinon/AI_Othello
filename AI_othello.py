import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.slim as slim

REPLAY_SIZE = 10000
BATCH_SIZE = 8
GAMMA = 1.0
BETA = 0.01

def dense(x, W, b, activation = 'relu', use_bn = True):
	x = tf.add(tf.matmul(x, W), b)
	if use_bn: x = tf.layers.batch_normalization(x)
	if activation == 'x': return x
	if activation == 'sigmoid': return tf.nn.sigmoid(x)
	if activation == 'relu': return tf.nn.relu(x)

def conv2D(x, W, b, strides = 1, activation = 'relu', use_bn = True):
	x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	if use_bn: x = tf.layers.batch_normalization(x)
	if activation == 'relu': return tf.nn.relu(x)

class Model:
	
	def __init__(self):
		self.replayMemory = []
		# Define Q networks
		self.weights = {
			'wc1': tf.get_variable('wc1', [3, 3, 3, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wc2': tf.get_variable('wc2', [3, 3, 16, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wc3': tf.get_variable('wc3', [3, 3, 16, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wc4': tf.get_variable('wc4', [3, 3, 16, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wd1': tf.get_variable('wd1', [8*8*16, 64], initializer = tf.contrib.layers.xavier_initializer()),
			'wd2': tf.get_variable('wd2', [64, 64], initializer = tf.contrib.layers.xavier_initializer())
		}
		self.biases = {
			'bc1': tf.get_variable('bc1', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bc2': tf.get_variable('bc2', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bc3': tf.get_variable('bc3', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bc4': tf.get_variable('bc4', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bd1': tf.get_variable('bd1', [64], initializer = tf.contrib.layers.xavier_initializer()),
			'bd2': tf.get_variable('bd2', [64], initializer = tf.contrib.layers.xavier_initializer()),
		}
		self.state = tf.placeholder(tf.float32, [None, 8, 8, 3])
		self.Q = self.forward(self.state, self.weights, self.biases)
		self.action = tf.placeholder(tf.float32, [None, 8, 8])
		self.Q_action = tf.reduce_sum(tf.multiply(self.Q, self.action), axis = [1, 2] )

		# Define target networks
		self.weights_T = {
			'wc1': tf.get_variable('wc1T', [3, 3, 3, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wc2': tf.get_variable('wc2T', [3, 3, 16, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wc3': tf.get_variable('wc3T', [3, 3, 16, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wc4': tf.get_variable('wc4T', [3, 3, 16, 16], initializer = tf.contrib.layers.xavier_initializer()),
			'wd1': tf.get_variable('wd1T', [8*8*16, 64], initializer = tf.contrib.layers.xavier_initializer()),
			'wd2': tf.get_variable('wd2T', [64, 64], initializer = tf.contrib.layers.xavier_initializer())
		}
		self.biases_T = {
			'bc1': tf.get_variable('bc1T', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bc2': tf.get_variable('bc2T', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bc3': tf.get_variable('bc3T', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bc4': tf.get_variable('bc4T', [16], initializer = tf.contrib.layers.xavier_initializer()),
			'bd1': tf.get_variable('bd1T', [64], initializer = tf.contrib.layers.xavier_initializer()),
			'bd2': tf.get_variable('bd2T', [64], initializer = tf.contrib.layers.xavier_initializer()),
		}

		self.state_T = tf.placeholder(tf.float32, [None, 8, 8, 3])
		self.Q_T = self.forward(self.state_T, self.weights_T, self.biases_T)
		self.Q_hat = tf.placeholder(tf.float32, [None]) # Same as R + gamma * max(Q_T)
		
		self.loss = tf.reduce_mean(tf.square(self.Q_hat - self.Q_action))
		self.regularizer = tf.Variable(0.0, trainable = False)
		
		for keys in self.weights:
			if keys == 'wd2': continue
			self.regularizer += tf.nn.l2_loss(self.weights[keys])
		
		self.regloss = tf.reduce_mean(self.loss + BETA * self.regularizer)
		
		self.global_step = tf.Variable(0, trainable = False)
		#self.learning_rate = tf.train.exponential_decay(0.01, self.global_step, 1000, 0.99, staircase = True)
		self.learning_rate = tf.Variable(0.001)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.regloss, global_step = self.global_step)
		
		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = 8
		config.inter_op_parallelism_threads = 8
		self.sess = tf.InteractiveSession(config = config)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(tf.global_variables())
		self.copyTargetNetwork()
		
	def forward(self, x, weights, biases):
		x = tf.reshape(x, [-1, 8, 8, 3])
		x = conv2D(x, weights['wc1'], biases['bc1'], activation = 'relu')
		x = conv2D(x, weights['wc2'], biases['bc2'], activation = 'relu')
		#x = conv2D(x, weights['wc3'], biases['bc3'], activation = 'relu')
		#x = conv2D(x, weights['wc4'], biases['bc4'], activation = 'relu')
		x = tf.reshape(x, [-1, 8*8*16])
		x = dense(x, weights['wd1'], biases['bd1'], activation = 'relu')
		x = dense(x, weights['wd2'], biases['bd2'], activation = 'sigmoid', use_bn = False)
		x = tf.reshape(x, [-1, 8, 8])
		return x

	def	copyTargetNetwork(self):
		for key in self.weights:
			tf.assign(self.weights_T[key], self.weights[key])
		for key in self.biases:
			tf.assign(self.biases_T[key], self.biases[key])
 
	def train(self):
		if len(self.replayMemory) < REPLAY_SIZE: return -1, -1, -1
		state, action, reward, nextState, isTerminal = [], [], [], [], []
		for _ in range(BATCH_SIZE):
			pos = random.randrange(0, REPLAY_SIZE)
			batch = self.replayMemory[pos]
			state.append(batch[0])
			action.append(batch[1])
			reward.append(batch[2])
			nextState.append(batch[3])
			isTerminal.append(batch[4])

		Q_T = self.Q_T.eval(session = self.sess, feed_dict = {self.state_T: state})
		Q_hat = []
		for i in range(BATCH_SIZE):
			if isTerminal[i] == True: Q_hat.append(reward[i])
			else: Q_hat.append(reward[i] + GAMMA * np.max(Q_T[i]))
		
		_, loss, regloss, learning_rate = self.sess.run([self.opt, self.loss, self.regloss, self.learning_rate], feed_dict = {self.state: state, self.Q_hat: Q_hat, self.action: action})
		return loss, regloss, learning_rate

	def save(self, name):
		self.saver.save(self.sess, name)

	def load(self, name):
		self.saver.restore(self.sess, name)
	
	def isMemoryFilled(self):
		if len(self.replayMemory) < REPLAY_SIZE: return False
		return True

	def addReplay(self, state, action, reward, nextState, isTerminated):
		if len(self.replayMemory) < REPLAY_SIZE:
			self.replayMemory.append([state, action, reward, nextState, isTerminated])
		else:
			pos = random.randrange(0, REPLAY_SIZE)
			self.replayMemory[pos] = [state, action, reward, nextState, isTerminated]
	
	def getQ(self, state):
		return self.Q_T.eval(session = self.sess, feed_dict = {self.state_T: [state]})[0]
	
	def getNextMove(self, state, validMove, epsilon):
		p = random.random()
		if p > epsilon:
			Q_T = self.Q_T.eval(session = self.sess, feed_dict = {self.state_T: [state]})[0]
			Q_max = -1e9
			npos = 0
			for pos in validMove:
				if Q_max < Q_T[pos]:
					Q_max = Q_T[pos]
					npos = pos
			return npos
		else:
			pos = random.randrange(0, len(validMove))
			return validMove[pos]
