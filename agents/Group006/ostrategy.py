from utils import print_board, count_moves


class OpeningStrategy:
    MAX_APPLIED_TURNS = 2

    def __init__(self):
        self.turn_count = 0
        self.direction = 1

    def is_apply(self):
        return self.turn_count < self.MAX_APPLIED_TURNS

    def next_move(self, board, red_turn=True):
        """
        Decide the next move given the input board.

        Input: 11*11 board list
        Output: positions (x, y)
        """
        self.turn_count += 1

        if self.turn_count == 1:
            if red_turn:
                return (1, 5)
            else:
                # allow to swap
                if self.whether_swap(board):
                    return (-1, -1)
                else:
                    return (5, 5)
        elif self.turn_count == 2:
            if red_turn:
                return self.virtual_connection(board, 1)
            else:
                if count_moves(board) == 1:
                    # the last player swapped
                    self.turn_count -= 1
                    return (5, 5)
                else:
                    return self.virtual_connection(board, -1)
        else:
            return self.virtual_connection(board, 1 if red_turn else -1)

    def whether_swap(self, board):
        # get the coordinates of the first red piece
        for x in range(11):
            for y in range(max(2, 5 - x // 2), min(9, 11 - (x % 2) - (x // 2))):
                if board[x][y] == 1:
                    return True
        return False

    def virtual_connection(self, board, color):
        # get the coordinates of the red and blue pieces, store them in a list
        red = []
        blue = []
        for i in range(11):
            for j in range(11):
                if board[i][j] == 1:
                    red.append((i, j))
                elif board[i][j] == -1:
                    blue.append((i, j))

        # decide grow direction
        if len(red) + len(blue) == 2:
            if red[0][0] > 5:
                self.direction = -1

        if color == 1:
            if self.direction == 1:
                # find the red piece that has largest x coordinate
                m_x_pos = max(red, key=lambda pos: pos[0])
            else:
                # find the red piece that has smallest x coordinate
                m_x_pos = min(red, key=lambda pos: pos[0])

            list_of_virtual_connections = [(m_x_pos[0] + 2 * self.direction, m_x_pos[1] - 1 * self.direction),
                                           (m_x_pos[0] + 1 * self.direction, m_x_pos[1] - 2 * self.direction),
                                           (m_x_pos[0] + 1 * self.direction, m_x_pos[1] + 1 * self.direction)]
            list_of_around = [(m_x_pos[0], m_x_pos[1] - 1 * self.direction),
                              (m_x_pos[0] + 1 * self.direction, m_x_pos[1] - 1 * self.direction),
                              (m_x_pos[0] + 1 * self.direction, m_x_pos[1]),
                              (m_x_pos[0], m_x_pos[1] + 1 * self.direction)]

            if (list_of_around[1] not in blue) and (list_of_around[2] not in blue) and (list_of_virtual_connections[0] not in blue):
                return list_of_virtual_connections[0]
            elif (list_of_around[0] not in blue) and (list_of_around[1] not in blue) and (list_of_virtual_connections[1] not in blue):
                return list_of_virtual_connections[1]
            elif (list_of_around[2] not in blue) and (list_of_around[3] not in blue) and (list_of_virtual_connections[2] not in blue):
                return list_of_virtual_connections[2]
            else:
                # choose a position in list_of_virtual_connections which is not occupied by blue
                for i in range(len(list_of_virtual_connections)):
                    if list_of_virtual_connections[i] not in blue:
                        return list_of_virtual_connections[i]

        else:
            # find the blue piece that has largest y coordinate
            m_y_pos = max(blue, key=lambda pos: pos[1])

            list_of_virtual_connections = [(m_y_pos[0] - 1, m_y_pos[1] + 2),
                                           (m_y_pos[0] - 2, m_y_pos[1] + 1),
                                           (m_y_pos[0] + 1, m_y_pos[1] + 1)]
            list_of_around = [(m_y_pos[0] - 1, m_y_pos[1]),
                              (m_y_pos[0] - 1, m_y_pos[1] + 1),
                              (m_y_pos[0], m_y_pos[1] + 1),
                              (m_y_pos[0] + 1, m_y_pos[1])]

            if (list_of_around[1] not in red) and (list_of_around[2] not in red) and (list_of_virtual_connections[0] not in red):
                return list_of_virtual_connections[0]
            elif (list_of_around[0] not in red) and (list_of_around[1] not in red) and (list_of_virtual_connections[1] not in red):
                return list_of_virtual_connections[1]
            elif (list_of_around[2] not in red) and (list_of_around[3] not in red) and (list_of_virtual_connections[2] not in red):
                return list_of_virtual_connections[2]
            else:
                # choose a position in list_of_virtual_connections which is not occupied by red
                for i in range(len(list_of_virtual_connections)):
                    if list_of_virtual_connections[i] not in red:
                        return list_of_virtual_connections[i]


if __name__ == "__main__":
    board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    os1 = OpeningStrategy()
    os2 = OpeningStrategy()
    os1_play_red = True
    while os1.is_apply() or os2.is_apply():
        if os1.is_apply():
            # x, y = os1.next_move(board, red_turn=os1_play_red)
            x, y = tuple(int(xx) for xx in input().split())
            if x == -1:
                os1_play_red = not os1_play_red
            else:
                board[x][y] = 1 if os1_play_red else -1
            print_board(board)

        if os2.is_apply():
            x, y = os2.next_move(board, red_turn=not os1_play_red)
            # x, y = tuple(int(xx) for xx in input().split())
            if x == -1:
                os1_play_red = not os1_play_red
            else:
                board[x][y] = 1 if not os1_play_red else -1
            print_board(board)
