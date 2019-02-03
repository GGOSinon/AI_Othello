import tensorflow as tf

def dense(x, W, b):
	return tf.add(tf.matmul(x, W), b)

def conv2D(x, W, b, strides = 1):
	x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

class Model:
	
	def __init__(self):
		# Define Q networks
		self.weights = {
			'wc1': tf.Variable(tf.random_normal([3, 3, 4, 32])),
			'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
			'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32])),
			'wc4': tf.Variable(tf.random_normal([3, 3, 32, 32])),
			'wd1': tf.Variable(tf.random_normal([8*8*32, 128])),
			'wd2': tf.Variable(tf.random_normal([128, 64]))
		}
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([32])),
			'bc2': tf.Variable(tf.random_normal([32])),
			'bc3': tf.Variable(tf.random_normal([32])),
			'bc4': tf.Variable(tf.random_normal([32])),
			'bd1': tf.Variable(tf.random_normal([128])),
			'bd2': tf.Variable(tf.random_normal([64]))
		}
		self.state = tf.placeholder(tf.float32, [None, 8, 8, 4])
		self.Q = self.forward(self.state, self.weights, self.biases)
		self.action = tf.placeholder(tf.float32, [None, 8, 8])
		self.Q_action = tf.reduce_sum(tf.multiply(self.Q, self.action), reduction_indices = 1)

		# Define target networks
		self.weights_T = {
			'wc1': tf.Variable(tf.random_normal([3, 3, 4, 32])),
			'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
			'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32])),
			'wc4': tf.Variable(tf.random_normal([3, 3, 32, 32])),
			'wd1': tf.Variable(tf.random_normal([8*8*32, 128])),
			'wd2': tf.Variable(tf.random_normal([128, 64]))
		}
		self.biases_T = {
			'bc1': tf.Variable(tf.random_normal([32])),
			'bc2': tf.Variable(tf.random_normal([32])),
			'bc3': tf.Variable(tf.random_normal([32])),
			'bc4': tf.Variable(tf.random_normal([32])),
			'bd1': tf.Variable(tf.random_normal([128])),
			'bd2': tf.Variable(tf.random_normal([64]))
		}
		self.state_T = tf.placeholder(tf.float32, [None, 8, 8, 4])
		self.Q_T = self.forward(self.state_T, self.weights_T, self.biases_T)
		
		self.Q_hat = tf.placeholder(tf.float32, [None]) # Same as R + gamma * max(Q_T)
		self.loss = tf.reduce_mean(tf.sqaure(tf.Q_hat, tf.Q_action))
		self.opt = tf.train.Adamoptimizer(learning_rate = 0.001).minimize(self.loss)

	def forward(self, x, weights, biases):
		x = tf.reshape(x, [-1, 8, 8, 4])
		x = conv2D(x, weights['wc1'], biases['bc1'])
		x = conv2D(x, weights['wc2'], biases['bc2'])
		x = conv2D(x, weights['wc3'], biases['bc3'])
		x = conv2D(x, weights['wc4'], biases['bc4'])
		x = tf.reshape(x, [-1, 8*8*32])
		x = dense(x, weights['wd1'], biases['bd1'])
		x = dense(x, weights['wd2'], biases['bd2'])
		x = tf.reshape(x, [-1, 8, 8])
		return x

	def	copyTargetNetwork(self):
		for key in self.weights:
			tf.assign(self.weights_T[key], self.weights[key])
		for key in self.biases:
			tf.assign(self.biases_T[key], self.biases[key])
 
	def train(self):
		#batch = random.sample(self.replayMemory, self.batch_size)
		state, action, reward, nextState, isTerminal = [], [], [], [], []
		for _ in range(BATCH_SIZE):
			pos = random.randrange(0, REPLAY_SIZE)
			batch = self.replayMemory[pos]
			state.append(batch[0])
			action.append(batch[1])
			reward.append(batch[2])
			nextState.append(batch[3])
			isTerminal.append(batch[4])

		Q_T = self.Q_T.eval(feed_dict = {self.state_T: state})
		Q_hat = []
		for i in range(self.batch_size):
			if isTerminal[i] == True: Q_hat.append(reward)
			else: Q_hat.append(reward + gamma * np.max(Q_T[i]))
		
		self.opt.run(feed_dict = {self.state: state, self.Q_hat: Q_hat, self.action: action})

	def addReplay(self, state, action, reward, nextState, isTerminated):
		if len(self.replayMemory) < REPLAY_SIZE:
			self.replayMemory.append([state, action, reward, nextState, isTerminated])
		else:
			pos = random.randrange(0, REPLAY_SIZE)
			self.replayMemory[pos] = [state, action, reward, nextState, isTerminated])

		
