import sys
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from utils import device, rotate_board
from vmodel import ValueModel
from ppredictor import PolicyPredictor


class ValueLearner:
    BOARD_SIZE = 11
    GAME_BATCH = 64
    EPOCH = 50000
    LEARNING_RATE = 0.001
    PERTURBATION = 0.05
    MAX_STARTING_STEPS = 60
    RANDOM_RATE = 0.5
    POLICY_CANDIDATES = 0

    PRETRAINED_PATH = ""
    PRETRAINED_EPOCH = 0
    SAVE_EVERY = 10
    CHECKPOINTS_FOLDER = "checkpoints/vmodel/"

    def __init__(self):
        self.evaluator = ValueModel()
        if self.PRETRAINED_PATH:
            self.evaluator.load_state_dict(torch.load(self.PRETRAINED_PATH, map_location=device))
            print(f'Loaded weights from "{self.PRETRAINED_PATH}".')
        else:
            self.initialize_weights()
        self.evaluator.train().to(device)
        self.optimizer = optim.Adam(self.evaluator.parameters(), lr=self.LEARNING_RATE)
        if self.POLICY_CANDIDATES > 0:
            self.predictor = PolicyPredictor()

        if not os.path.exists(self.CHECKPOINTS_FOLDER):
            os.mkdir(self.CHECKPOINTS_FOLDER)

    def initialize_weights(self):
        for m in self.evaluator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def train(self):
        """
        Train evaluator.
        """
        # number of training
        for e in range(self.PRETRAINED_EPOCH, self.EPOCH):
            print(f"Epoch: {e + 1} / {self.EPOCH}")
            start_time = time.time()
            # compute loss by self-play
            losses = self.self_play()
            # check nan
            if losses != losses:
                print("Losses is nan. Stop training.")
                break
            # calculate time
            end_time = time.time()
            print(f"Losses: {losses}, Time: {end_time - start_time:.2f}s")
            # save model
            if (e + 1) % self.SAVE_EVERY == 0:
                save_path = os.path.join(self.CHECKPOINTS_FOLDER, f"ep{e+1}_gb{self.GAME_BATCH}_loss{losses:.4f}.pth")
                torch.save(self.evaluator.state_dict(), save_path)
                print(f'Saved checkpoint to "{save_path}".')
                ValueLearner.PRETRAINED_PATH = save_path
                ValueLearner.PRETRAINED_EPOCH = e + 1
            sys.stdout.flush()

    def self_play(self):
        """
        Self play games in batches, and update weights in each turn.

        Output: total losses
        """
        losses = 0
        # initialise board
        print("Initialising game boards...", end="\r", flush=True)
        b_board, b_t = self.get_starting_board()
        b_ended = torch.logical_or(self.has_ended(b_board, b_t, player=1), self.has_ended(b_board, b_t, player=-1))
        if b_ended.all():
            return losses
        b_not_ended = b_ended.logical_not()
        # update board
        b_board = b_board[b_not_ended]
        # play until end
        depth = 0
        final_v1, final_v2 = [], []
        while b_not_ended.any():
            depth += 1
            print(f"Games in progress: {b_board.shape[0]}/{self.GAME_BATCH}, Depth: {depth}...", end="\r", flush=True)
            b_t += 1
            # decide move
            b_board, b_v1 = self.next_move(b_board, allow_swap=(b_t == 2))
            # decide ending
            b_ended = self.has_ended(b_board, b_t, player=1)
            b_not_ended = b_ended.logical_not()
            # swap turn
            b_board = rotate_board(b_board)
            # evaluate opponent's board
            b_v2 = self.evaluator(b_board)
            # update board
            b_board = b_board[b_not_ended]

            # compute reward target in next simulated turn
            with torch.no_grad():
                b_tmp_board, b_v3_nograd = self.next_move(b_board, allow_swap=(b_t + 1 == 2))
                b_v4_nograd = self.evaluator(rotate_board(b_tmp_board))

            # compute loss
            loss = torch.tensor(0.0).to(device)
            # symmetry loss
            # loss += self.loss(b_v1, 1 - b_v2)
            # ending loss
            loss += self.loss(b_v1[b_ended], 1, end=True)
            loss += self.loss(b_v2[b_ended], 0, end=True)
            # delayed reward loss
            loss += self.loss(b_v2[b_not_ended], b_v3_nograd)
            loss += self.loss(b_v1[b_not_ended], b_v4_nograd)

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses += loss.item()

            # store statistics
            final_v1 += b_v1[b_ended, 0].tolist()
            final_v2 += b_v2[b_ended, 0].tolist()

        print(f"Average v1: {sum(final_v1) / len(final_v1):.4f}, Average v2: {sum(final_v2) / len(final_v2):.4f}")
        return losses / depth

    def get_starting_board(self):
        """
        Generate starting game boards in batches.

        Output: b*11*11 tensor, b tensor (number of played steps for each board)
        """
        b_T = torch.randint(1, self.MAX_STARTING_STEPS + 1, (self.GAME_BATCH,))
        b_board = torch.zeros(self.GAME_BATCH, self.BOARD_SIZE, self.BOARD_SIZE, dtype=torch.int8).to(device)
        # play the first move
        for b in range(self.GAME_BATCH):
            x0, y0 = [random.choice([0, 1, self.BOARD_SIZE - 1, self.BOARD_SIZE - 2]) for _ in range(2)]
            b_board[b, x0, y0] = 1
        b_board = rotate_board(b_board)
        # play middle moves
        for t in range(2, b_T.max().item()):
            b_mask = (b_T > t)
            b_board[b_mask], _ = self.next_move(b_board[b_mask], allow_swap=(t == 2), pick_random=self.RANDOM_RATE, policy_candidates=self.POLICY_CANDIDATES, require_score=False)
            b_board[b_mask] = rotate_board(b_board[b_mask])
        # play the last random move
        b_mask = (b_T > 1)
        b_board[b_mask], _ = self.next_move(b_board[b_mask], allow_swap=(b_T == 2), pick_random=True, require_score=False)
        b_board[b_mask] = rotate_board(b_board[b_mask])
        return b_board, b_T

    def next_move(self, b_board, allow_swap=False, pick_random=False, policy_candidates=0, require_score=True):
        """
        Decide the next move given game board in batches.

        Input: b*11*11 tensor
        Output: b*11*11 new boards of next step, b*1 evaluation scores
        """
        batch_size = b_board.size(0)
        if isinstance(allow_swap, bool):
            allow_swap = torch.tensor(allow_swap).repeat(batch_size)
        pick_random = float(pick_random)
        pick_random = torch.rand(batch_size) < pick_random

        b_next_board = torch.empty_like(b_board)
        if require_score:
            b_score = torch.empty(batch_size, 1).to(device)

        can_boards_list = []
        for b in range(batch_size):
            board = b_board[b]
            if policy_candidates == 0:
                # find out coordinates of all the possible positions
                x, y = torch.where(board.to('cpu') == 0)
                if allow_swap[b]:
                    x = torch.cat((x, torch.tensor([-1])))
                    y = torch.cat((y, torch.tensor([-1])))
            else:
                # find out possible moves from policy predictor
                policy_moves = self.predictor.next_move(board.tolist(), red_turn=True, allow_swap=allow_swap[b], k=policy_candidates)
                policy_moves = torch.tensor(policy_moves)
                x, y = policy_moves[:, 0], policy_moves[:, 1]

            if pick_random[b]:
                # get a random move
                rand_idx = random.randint(0, x.shape[0] - 1)
                next_board = board.clone()
                if x[rand_idx] == -1:
                    next_board = rotate_board(next_board)
                else:
                    next_board[x[rand_idx], y[rand_idx]] = 1
                b_next_board[b] = next_board
                if require_score:
                    can_boards_list.append(next_board.unsqueeze(0))
            else:
                # generate candidate boards
                can_boards = board.repeat(x.shape[0], 1, 1)
                for i in range(x.shape[0]):
                    if x[i] == -1:
                        can_boards[i] = rotate_board(can_boards[i])
                    else:
                        can_boards[i, x[i], y[i]] = 1
                can_boards_list.append(can_boards)

        if can_boards_list:
            # evaluate candidate boards
            all_can_boards = torch.cat(can_boards_list, dim=0)
            if require_score:
                all_scores = self.evaluator(all_can_boards)
            else:
                with torch.no_grad():
                    all_scores = self.evaluator(all_can_boards)
            # add small perturbation to evaluated scores
            all_p_scores = all_scores + (torch.rand_like(all_scores) * (2 * self.PERTURBATION) - self.PERTURBATION)
            # choose the best move for non-random boards
            can_boards_idx = 0
            scores_idx = 0
            for b in range(batch_size):
                if not pick_random[b]:
                    can_boards = can_boards_list[can_boards_idx]
                    can_num = can_boards.shape[0]
                    best_idx = scores_idx + torch.argmax(all_p_scores[scores_idx: scores_idx + can_num])
                    b_next_board[b] = all_can_boards[best_idx]
                    if require_score:
                        b_score[b] = all_scores[best_idx]
                    can_boards_idx += 1
                    scores_idx += can_num
                else:
                    if require_score:
                        b_score[b] = all_scores[scores_idx]
                        can_boards_idx += 1
                        scores_idx += 1

        return b_next_board, b_score if require_score else None

    def loss(self, b_v1, b_v2, end=False):
        """
        Calculate the sum of losses of evaluation scores.
        """
        # if end:
        #     if b_v2 == 0:
        #         l = -torch.log(1 - b_v1)
        #     else:
        #         l = -torch.log(b_v1)
        # else:
        #     l = torch.square(b_v1 - b_v2)
        # return torch.sum(l)
        if b_v1.numel() > 0:
            return torch.mean(torch.square(b_v1 - b_v2))
        else:
            return 0

    def has_ended(self, b_board, b_t, player=1):
        """
        Check whether the game ended for given player (1/-1) on game board in batches.

        Input: b*n*n tensor (n = 11), b tensor (counting of played steps), player (1/-1)
        Output: b booleans tensor
        """
        b_ended = torch.zeros(b_board.shape[0], dtype=torch.bool)
        # check each board
        for b in range(b_board.shape[0]):
            board = b_board[b]
            # fast check by played steps
            if b_t[b] <= (board.shape[0] - 1) * 2:
                continue
            visited = torch.zeros(board.shape, dtype=torch.bool)
            if player == 1:
                # iterate over first row
                for j in range(board.shape[1]):
                    if not visited[0, j] and board[0, j] == player:
                        if self.dfs_ended(0, j, board, visited, player):
                            b_ended[b] = True
                            break
            else:
                # iterate over first column
                for i in range(board.shape[0]):
                    if not visited[i, 0] and board[i, 0] == player:
                        if self.dfs_ended(i, 0, board, visited, player):
                            b_ended[b] = True
                            break
        return b_ended

    def dfs_ended(self, x, y, board, visited, player):
        """"
        A recursive DFS that iterates through connected tiles of player's colour until it reaches bottom/right on a board.

        Output: boolean
        """
        if player == 1:
            # if reach the bottom, return True
            if x == board.shape[0] - 1:
                return True
        else:
            # if reach the right, return True
            if y == board.shape[1] - 1:
                return True
        visited[x, y] = True
        # visit neighbours
        neighbours = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        for dx, dy in neighbours:
            x_n = x + dx
            y_n = y + dy
            if x_n >= 0 and x_n < board.shape[0] and y_n >= 0 and y_n < board.shape[1]:
                if not visited[x_n, y_n] and board[x_n, y_n] == player:
                    if self.dfs_ended(x_n, y_n, board, visited, player):
                        return True
        return False


if __name__ == "__main__":
    while True:
        learner = ValueLearner()
        learner.train()
