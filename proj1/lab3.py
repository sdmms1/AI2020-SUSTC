import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, -1], [1, -1], [-1, 1]]
FDIRECTIONS = [[1, 0], [0, 1], [1, 1], [1, -1]]
WEIGHT = np.array([[500, -500, 100, 50, 50, 100, -500, 500],
                   [-500, -800, 50, 10, 10, 50, -800, -500],
                   [100, 50, 200, 5, 5, 200, 50, 100],
                   [50, 10, 5, 5, 5, 5, 10, 50],
                   [50, 10, 5, 5, 5, 5, 10, 50],
                   [100, 50, 200, 5, 5, 200, 50, 100],
                   [-500, -800, 50, 10, 10, 50, -800, -500],
                   [500, -500, 100, 50, 50, 100, -500, 500]])
INF = float("inf")


# don't change the class name
class AI(object):
    NUM = 0

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
                    temp = self.check(chessboard, x, y, DIRECTIONS[i], color)
                    if temp is not None:
                        out.add(temp)
        return list(out)

    def move(self, chessboard, place, color):
        init_x, init_y = place[0], place[1]
        newboard = chessboard.copy()
        newboard[init_x][init_y] = color
        reverse = list()
        opponent = -1 * color

        for d in DIRECTIONS:
            reverse.clear()
            x = init_x + d[0]
            y = init_y + d[1]
            if (x not in range(self.chessboard_size)) | (y not in range(self.chessboard_size)):
                continue

            while newboard[x][y] == opponent:
                newboard[x][y] *= -1
                reverse.append((x, y))
                x += d[0]
                y += d[1]
                if (x not in range(self.chessboard_size)) | (y not in range(self.chessboard_size)):
                    break

            if (x in range(self.chessboard_size)) & (y in range(self.chessboard_size)):
                if newboard[x][y] == color:
                    continue

            for e in reverse:
                newboard[e[0]][e[1]] *= -1

        # print(newboard)
        return newboard

    def color_num(self, chessboard, color):
        return len(np.where(chessboard == color)[0])

    def alpha_beta_search(self, chessboard, total_layer, current_layer, a_b):
        AI.NUM += 1
        alpha = a_b[0]
        beta = a_b[1]

        # check if the game is finished
        empty = self.color_num(chessboard, COLOR_NONE)
        if empty == 0:
            return alpha, beta, self.final_evaluate(chessboard)

        if current_layer >= total_layer:
            return alpha, beta, self.evaluate(chessboard, current_layer, empty)

        if current_layer % 2 == 1:
            # odd layer, use alpha, update beta
            v = INF
            available = self.get_avail(chessboard, self.color * -1)

            if not available:
                available = self.get_avail(chessboard, self.color)
                if not available:
                    # game over
                    return alpha, beta, self.final_evaluate(chessboard)
                else:
                    # skip
                    v = -INF
                    for e in available:
                        value = self.alpha_beta_search(self.move(chessboard, e, self.color),
                                                       total_layer, current_layer + 2, (alpha, beta))
                        v = max(value[2], v)
                        alpha = max(alpha, v)
                        if beta < alpha:
                            return alpha, beta, v
            else:
                for e in available:
                    value = self.alpha_beta_search(self.move(chessboard, e, self.color * -1),
                                                   total_layer, current_layer + 1, (alpha, beta))
                    v = min(value[2], v)
                    beta = min(beta, v)
                    if beta < alpha:
                        return alpha, beta, v
        else:
            # even layer, use beta, update alpha
            v = -INF
            available = self.get_avail(chessboard, self.color)

            if not available:
                available = self.get_avail(chessboard, self.color * -1)
                if not available:
                    # game over
                    return alpha, beta, self.final_evaluate(chessboard)
                else:
                    v = INF
                    # skip
                    for e in available:
                        value = self.alpha_beta_search(self.move(chessboard, e, self.color * -1),
                                                       total_layer, current_layer + 2, (alpha, beta))
                        v = min(value[2], v)
                        beta = min(beta, v)
                        if beta < alpha:
                            return alpha, beta, v
            else:
                for e in available:
                    value = self.alpha_beta_search(self.move(chessboard, e, self.color),
                                                   total_layer, current_layer + 1, (alpha, beta))
                    v = max(value[2], v)
                    alpha = max(alpha, v)
                    if beta < alpha:
                        return alpha, beta, v

        return alpha, beta, v

    # The input is current chessboard.
    def go(self, chessboard):
        # initialize
        start = time.time()
        self.candidate_list.clear()
        temp_list = list()
        self.candidate_list = self.get_avail(chessboard, self.color)
        empty = self.color_num(chessboard, COLOR_NONE)
        alpha, beta = -INF, INF
        # ==================================================================
        if empty < 10:
            for e in self.candidate_list:
                value = self.alpha_beta_search(self.move(chessboard, e, self.color), 14, 1, (alpha, beta))
                print(e, value)
                if value[2] > alpha:
                    alpha = value[2]
                    temp_list.append(e)
        elif empty > 56:
            for e in self.candidate_list:
                value = self.alpha_beta_search(self.move(chessboard, e, self.color), 2, 1, (alpha, beta))
                print(e, value)
                if value[2] > alpha:
                    alpha = value[2]
                    temp_list.append(e)
        else:
            for e in self.candidate_list:
                value = self.alpha_beta_search(self.move(chessboard, e, self.color), 2, 1, (alpha, beta))
                print(e, value)
                if value[2] > alpha:
                    alpha = value[2]
                    temp_list.append(e)
                elif value[2] == alpha:
                    if random.randint(0, 1):
                        temp_list.append(e)
        # ==================================================================
        if temp_list:
            self.candidate_list.append(temp_list[len(temp_list) - 1])

        end = time.time()
        print(self.candidate_list)
        print(empty, end - start, AI.NUM)

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will choose the last element of the candidate_list as the position you choose
        # If there is no valid position, you must return a empty list.

    def final_evaluate(self, chessboard):
        w = self.color_num(chessboard, COLOR_WHITE)
        b = self.color_num(chessboard, COLOR_BLACK)
        return (w - b) * 1000000 * self.color

    def evaluate(self, chessboard, current_layer, empty):
        # empty in (5, 60)
        mobility = self.get_mobility(chessboard, self.color * ((-1) ** current_layer))
        board = self.get_board_weight(chessboard)
        stability = self.get_stable(chessboard)

        if empty > 40:
            m_weight, b_weight, s_weight = 30, 1, 1500
        elif empty > 15:
            m_weight, b_weight, s_weight = 200, 1, 600
        else:
            m_weight, b_weight, s_weight = 300, 1, 600
        value = mobility * m_weight + board * b_weight + stability * s_weight
        print(chessboard)
        print(mobility, board, stability, value)
        return value * self.color

    def get_mobility(self, chessboard, color):
        available = self.get_avail(chessboard, color)
        return len(available) * color

    def get_board_weight(self, chessboard):
        weight = WEIGHT
        if chessboard[0][0] != COLOR_NONE:
            weight[0][0], weight[0][1], weight[1][0], weight[1][1] = 1000, -80, -80, -80
        if chessboard[7][0] != COLOR_NONE:
            weight[7][0], weight[7][1], weight[6][0], weight[6][1] = 1000, -80, -80, -80
        if chessboard[0][7] != COLOR_NONE:
            weight[0][7], weight[0][6], weight[1][7], weight[1][6] = 1000, -80, -80, -80
        if chessboard[7][7] != COLOR_NONE:
            weight[7][7], weight[7][6], weight[6][7], weight[6][6] = 1000, -80, -80, -80
        return np.sum(chessboard * weight)

    def get_stable(self, chessboard):
        queue = list()
        stable_map, count_map = np.zeros((8, 8)), np.zeros((8, 8))
        for c in ((0, 0), (0, 7), (7, 0), (7, 7)):
            if chessboard[c[0]][c[1]]:
                queue.append(c)

        for i in range(8):
            x, y = 0, i
            while not self.is_edge((x, y)):
                if not chessboard[x][y]:
                    while x > 0:
                        count_map[x][y] -= 1
                        x -= 1
                    break
                count_map[x][y] += 1
                x += 1

            x, y = i, 0
            while not self.is_edge((x, y)):
                if not chessboard[x][y]:
                    while y > 0:
                        count_map[x][y] -= 1
                        y -= 1
                    break
                count_map[x][y] += 1
                y += 1

        for i in range(1, 8):
            x, y = i, 0
            while not self.is_edge((x, y)):
                if not chessboard[x][y]:
                    while y > 0:
                        count_map[x][y] -= 1
                        x += 1
                        y -= 1
                    break
                count_map[x][y] += 1
                x -= 1
                y += 1

            x, y = i, 0
            while not self.is_edge((x, y)):
                if not chessboard[x][y]:
                    while y > 0:
                        count_map[x][y] -= 1
                        x -= 1
                        y -= 1
                    break
                count_map[x][y] += 1
                x += 1
                y += 1

            x, y = i, 7
            while not self.is_edge((x, y)):
                if not chessboard[x][y]:
                    while y < 8:
                        count_map[x][y] -= 1
                        x += 1
                        y += 1
                    break
                count_map[x][y] += 1
                x -= 1
                y -= 1

            x, y = i, 7
            while not self.is_edge((x, y)):
                if not chessboard[x][y]:
                    while y < 8:
                        count_map[x][y] -= 1
                        x -= 1
                        y += 1
                    break
                count_map[x][y] += 1
                x += 1
                y -= 1

        for i in range(1, 7):
            for j in range(1, 7):
                if count_map[i][j] == 4:
                    queue.append((i, j))
                    stable_map[i][j] = chessboard[i][j]
        # print(count_map)

        while len(queue):
            v = queue.pop(0)
            if self.check_stable(v, chessboard, stable_map):
                stable_map[v[0]][v[1]] = chessboard[v[0]][v[1]]
                queue.extend(self.get_neighbor(v, stable_map))
        print(stable_map)
        return self.color_num(stable_map, COLOR_WHITE) - self.color_num(stable_map, COLOR_BLACK)

    def get_neighbor(self, place, stable_map):
        neibours = []
        for d in DIRECTIONS:
            x, y = place[0] - d[0], place[1] - d[1]
            if self.is_edge((x, y)):
                continue
            if stable_map[x][y] == 0:
                neibours.append((x, y))
        return neibours

    def is_edge(self, place):
        return place[0] not in range(8) or place[1] not in range(8)

    def is_stable(self, place, stable_map, color):
        x, y = place[0], place[1]
        if stable_map[x][y] == 0:
            return 0
        else:
            return 1 if stable_map[x][y] == color else -1

    def check_stable(self, place, chessboard, stable_map):
        color = chessboard[place[0]][place[1]]
        for d in FDIRECTIONS:
            a, b = (place[0] + d[0], place[1] + d[1]), (place[0] - d[0], place[1] - d[1])
            if self.is_edge(a) or self.is_edge(b):
                continue
            if self.is_stable(a, stable_map, color) == 1 or \
                    self.is_stable(b, stable_map, color) == 1:
                continue
            if self.is_stable(a, stable_map, color) == -1 and \
                    self.is_stable(b, stable_map, color) == -1:
                continue
            return False

        return True
