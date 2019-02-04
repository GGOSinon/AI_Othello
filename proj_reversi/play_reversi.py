from board_reversi import Board
from AI_reversi import Model
import numpy as np
#import matplotlib.pyplot as plt

'''
plt.ion()

def plot_realtime(losses):
	plt.figure(2)
	plt.clf()
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Loss')
	plt.plot(losses)
	plt.pause(0.001)  # pause a bit so that plots are updated
'''
board = Board(num_hole = 6, onGUI = False)
model = Model()
BLACK, WHITE = 1, 2
DISPLAY_RATE = 100
COPY_RATE = 100

player = BLACK
cnt_data = 0
epsilon = 1.0
tot_loss = 0
tot_regloss = 0
losses = []

while True:
	cnt_data += 1
	state = board.get_state(player)
	validMove = board.getValidMove(player)
	action = model.getNextMove(state, validMove, epsilon)
	array_action = np.zeros((8, 8))
	array_action[action] = 1
	board.move(player, action[0], action[1])
	nextState = board.get_state(player)
	if board.isTerminated():
		winner = board.getWinner()
		if winner == player:
			reward = 1
		else:
			reward = -1
		model.addReplay(state, array_action, reward, nextState, True)
		player = BLACK
		board.restart()
		continue
	model.addReplay(state, array_action, 0, nextState, False)
	if player == BLACK: player = WHITE
	else: player = BLACK
	if len(board.getValidMove(player)) == 0:
		if player == BLACK: player = WHITE
		else: player = BLACK
	if cnt_data >= 10000:
		loss, regloss, learning_rate = model.train()
		tot_loss += loss
		tot_regloss += regloss
		if cnt_data % DISPLAY_RATE == 0:
			mean_loss = tot_loss/DISPLAY_RATE
			mean_regloss = tot_regloss/DISPLAY_RATE
			#losses.append(tot_loss/DISPLAY_RATE)
			tot_loss = 0
			tot_regloss = 0
			epsilon *= 0.999
			print("epsilon = %.3f lr = %.5f loss = %.3f regloss = %.3f" % (epsilon, learning_rate, mean_loss, mean_regloss))
			
		if cnt_data % COPY_RATE == 0:
			model.copyTargetNetwork()
		#plot_realtime(losses)
	#print(validMove)
	#s = input()
	

