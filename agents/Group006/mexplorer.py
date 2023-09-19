import random
import time
import copy
from mnode import Node
from utils import print_board
from ppredictor import PolicyPredictor


class MCTSExplorer:
    TIME_BUDGET = 9
    POLICY_CANDIDATES = 10

    def __init__(self, board, to_play):
        self.root_state = copy.deepcopy(board)
        self.root_node = Node(to_play)
        self.predictor = PolicyPredictor()

    def make_move(self, move):
        """
        Make a move on the real chess board and update the root of the tree.
        """
        move = tuple(move)
        self.root_node = self.root_node.children.get(move, Node(self.root_node.player))
        self.root_node.parent = None
        self.play(self.root_state, move, self.root_node.player)

    def best_move(self):
        """
        Return the current best move based on performed searching.
        """
        # Return None if the game is over
        if self.has_ended(self.root_state, self.root_node.player):
            return None
        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root_node.children.values(), key=lambda n: n.times).times
        max_nodes = [n for n in self.root_node.children.values() if n.times == max_value]
        bestchild = random.choice(max_nodes)
        return bestchild.move

    def search(self):
        """
        Search and update the search tree for a specified amount of time in seconds.

        Output: number of simulations performed
        """
        start_time = time.time()
        num_simulation = 0
        while time.time() - start_time < self.TIME_BUDGET:
            node, state = self.select_node()
            outcome = self.simulation(state, node.to_play)
            self.backup(node, node.to_play, outcome)
            num_simulation += 1
        return num_simulation

    def select_node(self):
        """
        This function search through the MCTS tree and expand children for one leaf node.

        Output: Node, current board situation for the node
        """
        node = self.root_node
        state = copy.deepcopy(self.root_state)
        # pick the most urgent child until reach a leaf node
        while node.children:
            max_value = max(node.children.values(), key=lambda n: n.update_value()).value
            max_nodes = [n for n in node.children.values() if n.value == max_value]
            node = random.choice(max_nodes)
            self.play(state, node.move, node.player)
        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if not self.has_ended(state, node.player):
            self.expand(node, state)
            node = random.choice(list(node.children.values()))
            self.play(state, node.move, node.player)
        return node, state

    def expand(self, node, state):
        """
        Generate the children of the passed leaf node based on the available moves in the passed game state and add them to the tree.

        Input : leaf node to expand, the board state for the node
        """
        # find out possible moves from policy predictor
        for move in self.predictor.next_move(state, (node.to_play == 1), k=self.POLICY_CANDIDATES):
            node.add_child(Node(node.player, move, node))

    def simulation(self, state, to_play):
        """
        Simulate the possible moves for current board state until one of the player wins.

        Input : current board state, next player to place the chess piece
        Output : winner
        """
        if self.has_ended(state, -to_play):
            return -to_play
        moves = self.get_available_moves(state)
        random.shuffle(moves)
        for move in moves:
            self.play(state, move, to_play)
            to_play *= -1
            if self.has_ended(state, -to_play):
                return -to_play

    def backup(self, node, to_play, outcome):
        """
        Update the node statistics on the path from the passed node to root to reflect the outcome of a randomly simulated playout.

        Input: the leaf node to start the backup from, the color to play at the leaf, the simulation outcome (winner color)
        """
        # reward is calculated for player who just played at the node and not the next player to play
        reward = 0 if outcome == to_play else 1
        while node is not None:
            node.times += 1
            node.reward += reward
            node = node.parent
            reward = 1 - reward

    def get_available_moves(self, board):
        """
        Find all available moves from current board

        Input: board
        Output: List of available position
        """
        rows, cols = len(board), len(board[0])
        moves = []
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def play(self, board, move, to_play):
        """
        Place a move for the player on current board

        Input: current board state, move tuple, player
        Output: Updated board state
        """
        if board[move[0]][move[1]] == 0:  # Check whether the cell is occupied
            board[move[0]][move[1]] = to_play
        else:
            raise ValueError("Cell occupied")

    def has_ended(self, board, player):
        """
        Check if current board state has winning route for last played player

        Input: current board state, current player
        Output: boolean
        """
        rows, cols = len(board), len(board[0])
        visited = [[False] * cols for _ in range(rows)]
        if player == 1:
            # iterate over first row
            for j in range(cols):
                if not visited[0][j] and board[0][j] == player:
                    if self.dfs_ended(0, j, board, visited, player):
                        return True
        else:
            # iterate over first column
            for i in range(rows):
                if not visited[i][0] and board[i][0] == player:
                    if self.dfs_ended(i, 0, board, visited, player):
                        return True
        return False

    def dfs_ended(self, x, y, board, visited, player):
        """"
        A recursive DFS that iterates through connected tiles of player's colour until it reaches bottom/right on a board.

        Output: boolean
        """
        rows, cols = len(board), len(board[0])
        if player == 1:
            # if reach the bottom, return True
            if x == rows - 1:
                return True
        else:
            # if reach the right, return True
            if y == cols - 1:
                return True
        visited[x][y] = True
        # visit neighbours
        neighbours = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        for dx, dy in neighbours:
            x_n = x + dx
            y_n = y + dy
            if x_n >= 0 and x_n < rows and y_n >= 0 and y_n < cols:
                if not visited[x_n][y_n] and board[x_n][y_n] == player:
                    if self.dfs_ended(x_n, y_n, board, visited, player):
                        return True
        return False


if __name__ == '__main__':
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

    explorer = MCTSExplorer(board, to_play=1)

    turn = 'mcts'
    while not explorer.has_ended(explorer.root_state, explorer.root_node.player):
        if turn == 'mcts':
            num_simulation = explorer.search()
            print(f'Simulation: {num_simulation}')
            best_move = explorer.best_move()
            print(f"Best move: {best_move}")
            explorer.make_move(best_move)
        else:
            move = tuple(int(xx) for xx in input().split())
            explorer.make_move(move)
        print_board(explorer.root_state)
        # turn = 'mcts' if turn == 'user' else 'user'
