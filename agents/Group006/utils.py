import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

skynet_dir = os.path.dirname(os.path.realpath(__file__))


def print_board(board):
    """
    Print the board
    'W' for 1, 'B' for -1, 'O' for 0
    """
    rows, cols = len(board), len(board[0])
    for i in range(rows):
        print(' ' * i, end='')
        for j in range(cols):
            if board[i][j] == 1:
                print('W', end=' ')
            elif board[i][j] == -1:
                print('B', end=' ')
            else:
                print('O', end=' ')
        print()


def count_moves(board):
    # count the number of pieces on the board
    rows, cols = len(board), len(board[0])
    count = 0
    for x in range(rows):
        for y in range(cols):
            if board[x][y] != 0:
                count += 1
    return count


def rotate_board(board):
    """
    Flip the board along main diagonal and swap all values.
    """
    board = board.transpose(-1, -2)
    board = board * -1
    return board


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 27 board patterns
        self.patterns = torch.tensor(list(product([-1, 0, 1], repeat=3)), dtype=torch.int8).to(device)

    def encodeBoard(self, boards):
        """
        Encode the game board.

        Input: b*n*n tensor (n = 11)
        Output: b*27*(n+1)*(n+1) tensor
        """
        b = boards.shape[0]
        n = boards.shape[-1] + 1
        # padding surroundings
        boards = F.pad(boards, (1, 1, 1, 1), "constant", 0)
        boards[:, [0, -1], 1:-1] = 1
        boards[:, 1:-1, [0, -1]] = -1
        # generate encoding board by sliding patterns
        encodings = torch.zeros(b, 27, n, n).to(device)
        for i in range(n):
            for j in range(n):
                # mask out the board area
                area = boards[:, i:i + 2, j:j + 2].reshape(b, 1, 4)[:, :, :3]
                patterns_ = self.patterns.repeat(b, 1, 1)
                # consider corner cases
                if i == 0 and j == 0:
                    patterns_[:, :, 0] = area[:, :, 0]
                elif i == 0 and j == n - 1:
                    patterns_[:, :, 1] = area[:, :, 1]
                elif i == n - 1 and j == 0:
                    patterns_[:, :, 2] = area[:, :, 2]
                # pattern matching
                encodings[:, :, i, j] = (area == patterns_).all(dim=-1)
        return encodings
