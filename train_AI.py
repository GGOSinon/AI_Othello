from board_othello import Board
from AI_othello import Model
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.show()

def plot_realtime(losses):
	plt.figure(1)
	plt.clf()
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Loss')
	plt.plot(losses)
	plt.pause(0.001)  # pause a bit so that plots are updated

board = Board(num_hole = 6, onGUI = False)
model = Model()
BLACK, WHITE = 1, 2

DISPLAY_RATE = 1000
COPY_RATE = 10000
SAVE_RATE = 10000
DECAY_RATE = 500

player = BLACK
cnt_data = 0
cnt_play = 0
epsilon = 1.0
tot_loss = 0
tot_regloss = 0
losses = []
old_state = [0, None, None]
old_array_action = [0, None, None]

#get_state(player0, player1) -> player0's Q when player1's turn
while True:
	if player == WHITE: opponent = BLACK
	else: opponent = WHITE
	cnt_data += 1
	state = board.get_state(player)
	validMove = board.getValidMove(player)

	if old_state[player] is not None: 
		model.addReplay(old_state[player], old_array_action[player], 0, state[player], False)
	
	action = model.getNextMove(state, validMove, epsilon)
	array_action = np.zeros((8, 8))
	array_action[action] = 1
	
	old_state[player] = state
	old_array_action[player] = array_action
	
	nextPlayer = board.move(action[0], action[1])

	if board.isTerminated():
		winner = board.getWinner()
		if winner == player:
			reward = 1
		if winner == opponent:
			reward = 0
		if winner == 0:
			reward = 0.5
		model.addReplay(old_state[player], old_array_action[player], reward, state, True)
		model.addReplay(old_state[opponent], old_array_action[opponent], 1 - reward, state, True)
		player = BLACK
		board.restart()
		cnt_play += 1
		continue
	
	player = nextPlayer

	if model.isMemoryFilled():
		loss, regloss, learning_rate = model.train()
		tot_loss += loss
		tot_regloss += regloss
		if cnt_data % DISPLAY_RATE == 0:
			mean_loss = tot_loss/DISPLAY_RATE
			mean_regloss = tot_regloss/DISPLAY_RATE
			losses.append(mean_loss)
			tot_loss = 0
			tot_regloss = 0
			print("Data %d(Game %d) : epsilon = %.3f lr = %.5f loss = %.3f regloss = %.3f" % (cnt_data, cnt_play, epsilon, learning_rate, mean_loss, mean_regloss))
			plot_realtime(losses)	
		if cnt_data % COPY_RATE == 0:
			model.copyTargetNetwork()
			print('Copyed')
		if cnt_data % SAVE_RATE == 0:
			model.save(name = 'model/AI_reversi-'+str(cnt_data//SAVE_RATE)+'.ckpt')
		if cnt_data % DECAY_RATE == 0:
			if epsilon > 0.1: epsilon *= 0.999
	#print(validMove)
	#s = input()
	

