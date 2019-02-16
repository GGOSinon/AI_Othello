from board_reversi import Board
from AI_reversi import Model
import numpy as np
from time import sleep

board = Board(num_hole = 6, onGUI = True)
model = Model()
model.load('model/AI_reversi-5.ckpt')
BLACK, WHITE = 1, 2

myPlayer, comPlayer = BLACK, WHITE
player = BLACK

while True:
	if player == comPlayer:
		state = board.get_state(player, player)
		validMove = board.getValidMove(player)
		action = model.getNextMove(state, validMove, 0)
		Q = model.getQ(state)
		print(Q)
	else:
		s = input()
		A = board.getPlayerAction()
		isValid = A[0]
		action = A[1], A[2]
		print(isValid, action)
		if isValid == False: continue

	array_action = np.zeros((8, 8))
	array_action[action] = 1
	nextPlayer = board.move(action[0], action[1])
	player = nextPlayer
	if board.isTerminated():
		winner = board.getWinner()
		if winner == myPlayer:
			print('WIN')
		else:
			print('LOST')
		player = BLACK
		board.restart()
		continue
	

