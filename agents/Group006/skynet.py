from ostrategy import OpeningStrategy
from mexplorer import MCTSExplorer
from vpredictor import ValuePredictor
from utils import count_moves


class Skynet:
    C_MIN = 0.15  # initial coefficient for MCTS to evaluation value
    C_MAX = 0.75
    TURN_TO_C_MAX = 40  # the turn when c reaches C_MAX

    def __init__(self):
        self.strategy = OpeningStrategy()

    def next_move(self, board, red_turn=True):
        if self.strategy.is_apply():
            # apply opening strategy
            move = self.strategy.next_move(board, red_turn=red_turn)
            if not self.strategy.is_apply():
                # initialise MCTS
                self.explorer = MCTSExplorer(board, to_play=1 if red_turn else -1)
                self.predictor = ValuePredictor()
            return move
        else:
            # MCTS search
            self.explorer.search()
            board = self.explorer.root_state
            children = self.explorer.root_node.children.values()
            moves = [child.move for child in children]
            # value evaluation
            scores = self.predictor.evaluate_moves(board, moves, red_turn=red_turn)
            # calculate overall scores
            max_value = float('-inf')
            for i, child in enumerate(children):
                score = self.overall_score(child, scores[i], board)
                if score > max_value:
                    max_value = score
                    move = child.move
            return move

    def make_move(self, move):
        if not self.strategy.is_apply():
            self.explorer.make_move(move)

    def overall_score(self, node, eval_score, board):
        moves_num = count_moves(board)
        c = min(self.C_MIN + (self.C_MAX - self.C_MIN) * (moves_num / self.TURN_TO_C_MAX), self.C_MAX)
        return c * (node.reward / node.times) + (1 - c) * eval_score
