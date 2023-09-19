from math import sqrt, log


class Node:

    def __init__(self, to_play, move=None, parent=None):
        self.times = 0  # count the times this node is visited
        self.reward = 0  # average reward from current position
        self.move = move  # the move that leads to this node
        self.to_play = to_play  # the next player to play
        self.player = -to_play  # the last played player
        self.parent = parent
        self.children = {}
        self.update_value()

    def add_child(self, child):
        """
        Add a child node to the children of this node.
        """
        self.children[child.move] = child

    def update_value(self, exploration=0.5):
        """
        Calculate the UCT value of this node relative to its parent with some level of exploration.
        Return calculated value.

        Exploration indicates how possible should we choose nodes that have yet to be thoroughly explored or nodes might have a high winning rate
        """
        if self.parent is None:
            self.value = None
        else:
            if self.times == 0:
                if exploration == 0:
                    self.value = 0  # Indicates we do not want to explore
                else:
                    self.value = float('inf')
            else:
                # exploitation + exploration
                self.value = self.reward / self.times + exploration * sqrt(2 * log(self.parent.times + 1) / self.times)
        return self.value
