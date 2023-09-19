import os
import torch
from utils import device, skynet_dir, rotate_board
from pmodel import PolicyModel


class PolicyPredictor:
    PRETRAINED_PATH = os.path.join(skynet_dir, "weights/pmodel/pweights.pth")

    def __init__(self):
        self.model = PolicyModel()
        if self.PRETRAINED_PATH:
            self.model.load_state_dict(torch.load(self.PRETRAINED_PATH, map_location=device))
            print(f'Loaded weights from "{self.PRETRAINED_PATH}"')
        self.model.eval().to(device)

    def next_move(self, board, red_turn=True, allow_swap=False, k=10):
        """
        Decide the next move given the input board.

        Input: 11*11 board list
        Output: a list of top k (if have) positions (x, y), or a swap (-1, -1)
        """
        # read in board
        board = torch.tensor(board, dtype=torch.int8).to(device)
        if not red_turn:
            board = rotate_board(board)
        board = board.unsqueeze(0)
        # make prediction
        with torch.no_grad():
            pred = self.model(board)
        # sort predictions
        sorted_indices = pred.sort(descending=True).indices[0]

        # choose top k predictions
        positions = []
        num = 0
        index = 0
        while num < k and index < sorted_indices.shape[0]:
            # get coordinates
            idx = sorted_indices[index].item()
            if idx == 121:
                x = y = -1
            else:
                x, y = idx // 11, idx % 11

            if x == -1:
                # skip if not allow swap
                if not allow_swap:
                    index += 1
                    continue
            else:
                # skip if the position is already occupied
                if board[0, x, y] != 0:
                    index += 1
                    continue

            # record coordinates
            if not red_turn:
                x, y = y, x
            positions.append((x, y))
            num += 1
            index += 1

        return positions


if __name__ == "__main__":
    predictor = PolicyPredictor()
    input_board = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [-1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    positions = predictor.next_move(input_board, red_turn=True, allow_swap=False, k=10)
    print(len(positions))
    print(positions)
