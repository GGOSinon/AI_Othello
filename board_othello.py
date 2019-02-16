from tkinter import *
import numpy as np
import random

HOLE = -1
BLACK = 1
WHITE = 2
xx = [-1,-1,-1,0,0,1,1,1]
yy = [-1,0,1,-1,1,-1,0,1]
isClicked = False
mouseX, mouseY = 0, 0

class Board:
	
	def __init__(self, num_hole, onGUI):
		self.num_hole = num_hole
		self.onGUI = onGUI
		if self.onGUI:
			self.boardSize = 500
			self.margin = 50
			self.rectSize = (self.boardSize - 2*self.margin)/8
			self.GUIroot = Tk()
			self.GUIcanvas = Canvas(self.GUIroot, width = self.boardSize, height = self.boardSize)
			self.GUIcanvas.pack()
			self.GUIcanvas.bind('<Button 1>', self.getMouse)
		self.restart()

	def drawBoard(self):
		isClicked = False
		self.GUIcanvas.delete('all')
		for i in range(9):
			x = self.margin + i * self.rectSize
			self.GUIcanvas.create_line(self.margin, x, self.boardSize - self.margin, x)
			self.GUIcanvas.create_line(x, self.margin, x, self.boardSize - self.margin)
		for i in range(8):
			for j in range(8):
				x = self.margin + i * self.rectSize
				y = self.margin + j * self.rectSize
				if self.board[i][j] == HOLE:
					self.GUIcanvas.create_rectangle(x, y, x + self.rectSize, y + self.rectSize, fill='#F00')
				if self.board[i][j] == BLACK:
					self.GUIcanvas.create_oval(x, y, x + self.rectSize, y + self.rectSize, fill = '#000')
				if self.board[i][j] == WHITE:
					self.GUIcanvas.create_oval(x, y, x + self.rectSize, y + self.rectSize, fill = '#FFF')
		valid_moves = self.getValidMove(self.player)
		for i, j in valid_moves:
			x = self.margin + i * self.rectSize
			y = self.margin + j * self.rectSize
			self.GUIcanvas.create_rectangle(x, y, x + self.rectSize, y + self.rectSize, fill = '#0F0')
		self.GUIcanvas.update()

	def getPlayerAction(self):
		global isClicked, mouseX, mouseY
		if isClicked == True:
			isClicked = False
			X = int((mouseX - self.margin) // self.rectSize)
			Y = int((mouseY - self.margin) // self.rectSize)
			valid_moves = self.getValidMove(self.player)
			if (X, Y) in valid_moves:
				return True, X, Y
			else:
				return False, X, Y
		else:
		 	return False, 0, 0

	def getMouse(self, eventorigin):
		global isClicked, mouseX, mouseY
		isClicked = True
		mouseX = eventorigin.x
		mouseY = eventorigin.y
		print(isClicked, mouseX, mouseY)	
		
	def restart(self):
		self.board_b = np.zeros((8, 8))
		self.board_w = np.zeros((8, 8))
		self.isHole = np.zeros((8, 8))
		self.board = np.zeros((8, 8))
		self.player = BLACK
		self.move(3, 3)
		self.player = WHITE
		self.move(4, 3)
		self.player = BLACK
		self.move(4, 4)
		self.player = WHITE
		self.move(3, 4)
		self.player = BLACK
		cnt = 0
		while cnt < self.num_hole:
			x = random.randrange(0, 8)
			y = random.randrange(0, 8)
			if self.board[x][y] != 0: continue
			self.board[x][y] = HOLE
			self.isHole[x][y] = True
			if self.onGUI: self.drawBoard()
			cnt += 1

	def get_state(self, player):
		if player == BLACK:
			state = np.stack([self.board_b, self.board_w, self.isHole], axis = 2)
		if player == WHITE:
			state = np.stack([self.board_w, self.board_b, self.isHole], axis = 2)
		return state


	def changePlayer(self):
		if self.player == BLACK: self.player = WHITE
		else: self.player = BLACK

	def move(self, x, y):
		self.board[x][y] = self.player
		if self.player == BLACK:
			self.board_b[x][y] = True
		if self.player == WHITE:
			self.board_w[x][y] = True
		self.turnover(self.player, x, y)
		self.changePlayer()
		if self.hasValidMove(self.player) == False: self.changePlayer()
		if self.onGUI:
			self.drawBoard()
		return self.player

	def turnover(self, player, x, y):
		tot_cnt = 0
		for dir in range(8):
			nx = x
			ny = y
			cnt = 0
			isValid = False
			while True:
				nx += xx[dir]
				ny += yy[dir]
				if nx<0 or nx>=8 or ny<0 or ny>=8: break
				if self.board[nx][ny] == 0 or self.board[nx][ny] == -1:
					break
				if self.board[nx][ny] == player:
					if cnt>0: isValid = True
					break
				cnt += 1
			if isValid:
				tot_cnt += cnt
				nx = x
				ny = y
				for _ in range(cnt):
					nx += xx[dir]
					ny += yy[dir]
					self.board[nx][ny] = player
					
		return tot_cnt

	def turnover_simulate(self, player, x, y):
		tot_cnt = 0
		isValid = False
		for dir in range(8):
			nx = x
			ny = y
			cnt = 0
			while True:
				nx += xx[dir]
				ny += yy[dir]
				if nx<0 or nx>=8 or ny<0 or ny>=8: break
				if self.board[nx][ny] == 0 or self.board[nx][ny] == -1:
					break
				if self.board[nx][ny] == player:
					if cnt>0: isValid = True
					break
				cnt += 1
			tot_cnt += cnt
		return isValid, tot_cnt

	def hasValidMove(self, player):
		if len(self.getValidMove(player)) == 0: return False
		return True

	def getValidMove(self, player):
		valid_moves = []
		for i in range(8):
			for j in range(8):
				if self.board[i][j] != 0: continue
				isValid, _ = self.turnover_simulate(player, i, j)
				if isValid: valid_moves.append((i,j))
		return valid_moves
	
	def isTerminated(self):
		if self.hasValidMove(BLACK) == False and self.hasValidMove(WHITE) == False: return True
		return False

	def getWinner(self):
		cnt_b = 0
		cnt_w = 0
		for i in range(8):
			for j in range(8):
				if self.board[i][j] == BLACK: cnt_b += 1
				if self.board[i][j] == WHITE: cnt_w += 1
		if cnt_b < cnt_w: return WHITE
		if cnt_b > cnt_w: return BLACK
		return 0
	

