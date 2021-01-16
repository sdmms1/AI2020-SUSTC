import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
directions = [[1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, -1], [1, -1], [-1, 1]]

# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your
        # decision.
        self.candidate_list = []

    def check(self, chessboard, x, y, direction, color):
        flag = False
        x, y = x + direction[0], y + direction[1]
        while (x in range(self.chessboard_size)) & (y in range(self.chessboard_size)):
            if chessboard[x][y] == color:
                return None
            elif chessboard[x][y] == COLOR_NONE:
                if flag:
                    return x, y
                else:
                    return None
            else:
                flag = True
                x, y = x + direction[0], y + direction[1]
        return None

    def get_avail(self, chessboard, color):
        out = set()
        for x in range(self.chessboard_size):
            for y in range(self.chessboard_size):
                if chessboard[x][y] != color:
                    continue

                for i in range(8):
                    temp = self.check(chessboard, x, y, directions[i], color)
                    if temp is not None:
                        out.add(temp)
        return list(out)

        # The input is current chessboard.

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        # idx = np.where(chessboard == COLOR_NONE)
        # idx = list(zip(idx[0], idx[1]))
        self.candidate_list = list(self.get_avail(chessboard))
        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will choose the last element of the candidate_list as the position you choose
        # If there is no valid position, you must return a empty list.
