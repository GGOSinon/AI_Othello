import tkinter
import numpy as np

Class Board:
	xx = [-1,-1,-1,0,0,1,1,1]
	yy = [-1,0,1,-1,1,-1,0,1]

	def __init__(self, num_hole, onGUI):
		self.board_b = np.zeros((8, 8))
		self.board_w = np.zeros((8, 8))
		self.isHole = np.zeros((8, 8))
		self.board = np.zeros((8, 8))
		self.onGUI = onGUI
		self.HOLE = -1
		self.BLACK = 1
		self.WHITE = 2
		for _ in range(num_hole):
			x = random.randrange(0, 8)
			y = random.randrange(0, 8)
			self.board[x][y] = self.HOLE
			self.isHole[x][y] = True

		if self.onGUI:
			self.boardSize = 500
			self.margin = 50
			self.rectSize = (self.boardSize - 2*self.margin)/8
			self.GUIroot = Tk()
			self.GUIcanvas = Canvas(self.GUIroot, width = self.boardSize, height = self.boardSize)
			self.GUIcanvas.pack()
			self.drawBoard()

	def drawBoard(self):
		for i in range(9):
			x = self.margin + i * self.rectSize
			self.GUIcanvas.create_line(self.margin, x, self.boardSize - self.margin, x)
			self.GUIcanvas.create_line(x, self.margin, x, self.boardSize - self.margin)
		for i in range(8):
			for j in range(8):
				x = self.margin + i * self.rectSize
				y = self.margin + j * self.rectSize
				if self.board[x][y] == self.HOLE:
					self.GUIcanvas.create_rectangle(x, y, x + self.rectSize, y + self.rectSize, fill='#F00')
				if self.board[x][y] == self.BLACK:
					self.GUIcanvas.create_oval(x, y, x + self.rectSize, y + self.rectSize, fill = '#000')
				if self.board[x][y] == self.WHITE:
					self.GUIcanvas.create_oval(x, y, x + self.rectSize, y + self.rectSize, fill = '#FFF')
		self.GUIcanvas.update()
	
	def move(self, player, x, y):
		self.board[x][y] = player
		if player == self.BLACK:
			self.board_b = True
		if player == self.WHITE:
			self.board_w = True
		self.turnover(player, x, y)
		self.drawBoard()

	def turnover(self, player, x, y):
		for dir in range(8):
			nx = x
			ny = y
			isValid = False
			while True:
				nx += xx[dir]
				ny += yy[dir]
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
					nx += xx[i]
					ny += yy[i]
					self.board[nx][ny] = player
					
		return tot_cnt

	def turnover_simulate(self, player, x, y):
		isValid = False
		for dir in range(8):
			nx = x
			ny = y
			while True:
				nx += xx[dir]
				ny += yy[dir]
				if self.board[nx][ny] == 0:
					break
				if self.board[nx][ny] == player:
					if cnt>0: isValid = True
					break
				cnt += 1
			tot_cnt += cnt
		return isValid, tot_cnt

	def getWinner(self):
		cnt_b = 0
		cnt_w = 0
		for i in range(8):
			for j in range(8):
				if self.board[i][j] == self.BLACK: cnt_b += 1
				if self.board[i][j] == self.WHITE: cnt_w += 1
		if cnt_b < cnt_w: return self.WHITE
		else: return self.BLACK
	

