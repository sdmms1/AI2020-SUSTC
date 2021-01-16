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
WEIGHT = [[250, -100, 20, 20, 20, 20, -100, 250],
          [-100, -120, 5, 5, 5, 5, -120, -100],
          [20, 5, 1, 1, 1, 1, 5, 20],
          [20, 5, 1, 1, 1, 1, 5, 20],
          [20, 5, 1, 1, 1, 1, 5, 20],
          [20, 5, 1, 1, 1, 1, 5, 20],
          [-100, -120, 5, 5, 5, 5, -120, -100],
          [250, -100, 20, 20, 20, 20, -100, 150]]


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
        # print(place)
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

    def search(self, chessboard, total_layer, current_layer, key_flag):
        AI.NUM += 1
        # check if the game is finished
        c = np.where(chessboard == COLOR_NONE)
        if len(c[0]) == 0:
            a = np.where(chessboard == COLOR_WHITE)
            b = np.where(chessboard == COLOR_BLACK)
            return (len(a[0]) - len(b[0])) * self.color * 10

        if current_layer == total_layer:
            return self.color * self.evaluate(chessboard, current_layer)

        # available = self.get_avail(chessboard, self.color)

        if current_layer % 2 == 1:
            # odd layer, the max value
            available = self.get_avail(chessboard, self.color * -1)
            key = 1000000
            for e in available:
                value = self.search(self.move(chessboard, e, self.color * -1), total_layer, current_layer + 1, key)
                if value < key_flag:
                    return value
                # print(e, value)
                key = value if value < key else key
            return key
        else:
            # even layer, take min value
            available = self.get_avail(chessboard, self.color)
            key = -1000000
            for e in available:
                value = self.search(self.move(chessboard, e, self.color), total_layer, current_layer + 1, key)
                if value > key_flag:
                    return value
                # print(e, value)
                key = value if value > key else key
            return key

    def search1(self, chessboard, total_layer, current_layer, a_b):
        AI.NUM += 1
        alpha = a_b[0]
        beta = a_b[1]
        # check if the game is finished
        c = np.where(chessboard == COLOR_NONE)
        if len(c[0]) == 0:
            a = np.where(chessboard == COLOR_WHITE)
            b = np.where(chessboard == COLOR_BLACK)
            c = (len(a[0]) - len(b[0])) * self.color * 10
            if current_layer % 2 == 1:
                return 0, c
            else:
                return c, 0

        if current_layer == total_layer:
            return 0, self.color * self.evaluate(chessboard, current_layer)  # notice here!!!

        # available = self.get_avail(chessboard, self.color)

        if current_layer % 2 == 1:
            # odd layer, use alpha, update beta
            available = self.get_avail(chessboard, self.color * -1)
            for e in available:
                value = self.search1(self.move(chessboard, e, self.color * -1),
                                     total_layer, current_layer + 1, (alpha, beta))
                if value[0] < beta:
                    beta = value[0]
                if beta < alpha:
                    return alpha, beta
        else:
            # even layer, use beta, update alpha
            available = self.get_avail(chessboard, self.color)
            for e in available:
                value = self.search1(self.move(chessboard, e, self.color),
                                     total_layer, current_layer + 1, (alpha, beta))
                if value[1] > alpha:
                    alpha = value[1]
                if beta < alpha:
                    return alpha, beta

        return alpha, beta

    def alpha_beta_search(self, chessboard, total_layer, current_layer, a_b):
        AI.NUM += 1
        alpha = a_b[0]
        beta = a_b[1]
        # check if the game is finished
        c = np.where(chessboard == COLOR_NONE)
        if len(c[0]) == 0:
            a = np.where(chessboard == COLOR_WHITE)
            b = np.where(chessboard == COLOR_BLACK)
            c = (len(a[0]) - len(b[0])) * self.color * 1000
            return alpha, beta, c

        if current_layer == total_layer:
            return alpha, beta, self.color * self.evaluate(chessboard, current_layer)

        # available = self.get_avail(chessboard, self.color)

        if current_layer % 2 == 1:
            # odd layer, use alpha, update beta
            v = float("inf")
            available = self.get_avail(chessboard, self.color * -1)
            for e in available:
                value = self.alpha_beta_search(self.move(chessboard, e, self.color * -1),
                                               total_layer, current_layer + 1, (alpha, beta))
                v = min(value[2], v)
                beta = min(beta, v)
                if beta < alpha:
                    return alpha, beta, v
        else:
            # even layer, use beta, update alpha
            v = -float("inf")
            available = self.get_avail(chessboard, self.color)
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
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        start = time.time()
        # AI_O = AI(self.chessboard_size, self.color * -1, self.time_out)
        # ==================================================================
        # Write your algorithm here
        temp_list = list()
        self.candidate_list = self.get_avail(chessboard, self.color)
        alpha = -float("inf")
        beta = float("inf")
        print(self.candidate_list)

        # ========================================================1
        # for e in self.candidate_list:
        #     value = self.search(self.move(chessboard, e, self.color), 4, 1, alpha)
        #     # print(e, value)
        #     if value > alpha:
        #         alpha = value
        #         temp_list.append(e)

        # ========================================================2
        # for e in self.candidate_list:
        #     value = self.search1(self.move(chessboard, e, self.color), 5, 1, (alpha, beta))
        #     print(e, value)
        #     if value[1] > alpha:
        #         alpha = value[1]
        #         temp_list.append(e)

        # ========================================================3
        for e in self.candidate_list:
            value = self.alpha_beta_search(self.move(chessboard, e, self.color), 6, 1, (alpha, beta))
            print(e, value)
            if value[1] > alpha:
                alpha = value[1]
                temp_list.append(e)

        if temp_list:
            self.candidate_list.append(temp_list[len(temp_list) - 1])

        end = time.time()
        print(end - start, len(np.where(chessboard == COLOR_NONE)[0]), AI.NUM)
        print(self.candidate_list)

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will choose the last element of the candidate_list as the position you choose
        # If there is no valid position, you must return a empty list.

    def evaluate(self, chessboard, current_layer):
        print(chessboard)
        motability = self.get_mobility(chessboard, self.color * (-1 ** current_layer))
        weight = self.get_board_weight(chessboard)
        stability = self.get_stable(chessboard)
        return motability * 5 + weight + stability * 20

    def get_mobility(self, chessboard, color):
        arr = self.get_avail(chessboard, color)
        flag = color + self.color
        return len(arr) if flag else len(arr) * -1

    def get_board_weight(self, chessboard):
        weight = WEIGHT
        if chessboard[0][0] != COLOR_NONE:
            weight[0][0], weight[0][1], weight[1][0], weight[1][1] = 30, 5, 5, 5
        if chessboard[7][0] != COLOR_NONE:
            weight[7][0], weight[7][1], weight[6][0], weight[6][1] = 30, 5, 5, 5
        if chessboard[0][7] != COLOR_NONE:
            weight[0][7], weight[0][6], weight[1][7], weight[1][6] = 30, 5, 5, 5
        if chessboard[7][7] != COLOR_NONE:
            weight[7][7], weight[7][6], weight[6][7], weight[6][6] = 30, 5, 5, 5
        return np.sum(chessboard * weight)

    def get_stable(self, chessboard):
        queue = list()
        stable_map = np.zeros((8, 8))
        for c in ((0, 0), (0, 7), (7, 0), (7, 7)):
            if chessboard[c[0]][c[1]]:
                queue.append(c)

        while len(queue):
            v = queue.pop(0)
            if self.check_stable(v, chessboard, stable_map):
                stable_map[v[0]][v[1]] = chessboard[v[0]][v[1]]
                queue.extend(self.get_neibour(v, chessboard, stable_map))

        a = np.where(chessboard == COLOR_WHITE)
        b = np.where(chessboard == COLOR_BLACK)
        # print(stable_map)
        return (len(a[0]) - len(b[0])) * self.color * 10


    def get_neibour(self, place, chessboard, stable_map):
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
