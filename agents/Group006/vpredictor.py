import os
import torch
from utils import device, skynet_dir, rotate_board
from vmodel import ValueModel


class ValuePredictor:
    PRETRAINED_PATH = os.path.join(skynet_dir, "weights/vmodel/vweights.pth")

    def __init__(self):
        self.evaluator = ValueModel()
        if self.PRETRAINED_PATH:
            self.evaluator.load_state_dict(torch.load(self.PRETRAINED_PATH, map_location=device))
            print(f'Loaded weights from "{self.PRETRAINED_PATH}".')
        self.evaluator.eval().to(device)

    def next_move(self, board, red_turn=True, allow_swap=False):
        """
        Decide the next move given the input board.

        Input: 11*11 board list
        Output: move position (x, y), or a swap (-1, -1)
        """
        board = torch.tensor(board, dtype=torch.int8).to(device)
        # swap colour if blue turn
        if not red_turn:
            board = rotate_board(board)
        # find out coordinates of all the possible positions
        x0, y0 = torch.where(board.to('cpu') == 0)
        if allow_swap:
            x0 = torch.cat((x0, torch.tensor([-1])))
            y0 = torch.cat((y0, torch.tensor([-1])))
        b = x0.shape[0]
        # generate candidate boards
        boards = board.repeat(b, 1, 1)
        for i in range(b):
            if x0[i] == -1:
                boards[i] = rotate_board(boards[i])
            else:
                boards[i, x0[i], y0[i]] = 1
        # evaluate candidate boards
        with torch.no_grad():
            scores = self.evaluator(boards)
        # choose the best move
        best_idx = scores.argmax()
        x, y = x0[best_idx], y0[best_idx]
        # swap back if blue turn
        if not red_turn:
            x, y = y, x
        return x.item(), y.item()

    def evaluate_moves(self, board, moves, red_turn=True):
        """
        Evaluate the value of a list of moves.

        Input: 11*11 board list, a list of moves (x, y)
        Output: a list of scores
        """
        board = torch.tensor(board, dtype=torch.int8).to(device)
        # swap colour if blue turn
        if not red_turn:
            board = rotate_board(board)
        # generate candidate boards
        b = len(moves)
        boards = board.repeat(b, 1, 1)
        for i in range(b):
            x, y = moves[i]
            if not red_turn:
                x, y = y, x
            if x == -1:
                boards[i] = rotate_board(boards[i])
            else:
                boards[i, x, y] = 1
        # evaluate candidate boards
        with torch.no_grad():
            scores = self.evaluator(boards)
        return scores[:, 0].tolist()


if __name__ == "__main__":
    predictor = ValuePredictor()
    input_board = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, -1, 0, 1, 0, 0, 0, -1, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
                   [-1, -1, -1, 1, 0, 0, 0, 0, -1, -1, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0],
                   [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    x, y = predictor.next_move(input_board, red_turn=True, allow_swap=False)
    print(x, y)
    scores = predictor.evaluate_moves(input_board, [(6, 3), (10, 4)], red_turn=True)
    print(scores)
