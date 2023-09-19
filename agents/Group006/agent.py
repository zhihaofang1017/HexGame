import socket
from skynet import Skynet


class SkynetAgent():
    """
    The Skynet hex agent.
    """

    HOST = "127.0.0.1"
    PORT = 1234

    COLOUR_MAP = {"R": 1, "B": -1}
    OPP_COLOUR = {"R": "B", "B": "R"}

    def __init__(self, board_size=11):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST, self.PORT))

        self.board_size = board_size
        self.board = []
        self.colour = ""

        self.skynet = Skynet()

    def run(self):
        """
        Read data until it receives an END message or the socket closes.
        """

        while True:
            data = self.s.recv(1024)
            if not data:
                break
            # print(f"{self.colour} {data.decode('utf-8')}", end="")
            if self.interpret_data(data):
                break

        self.s.close()
        # print(f"Naive agent {self.colour} terminated")

    def interpret_data(self, data):
        """
        Check the type of message and responds accordingly.
        Return True if the game ended, False otherwise.
        """

        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        # print(messages)
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [[0] * self.board_size for _ in range(self.board_size)]

                if self.colour == "R":
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.OPP_COLOUR[self.colour]
                    if s[3] == self.colour:
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.COLOUR_MAP[self.OPP_COLOUR[self.colour]]
                    self.skynet.make_move((action[0], action[1]))

                    self.make_move()

        return False

    def make_move(self):
        """
        Decide a move by predictor.
        """

        # print(f"{self.colour} making move")
        x, y = self.skynet.next_move(self.board, red_turn=(self.colour == "R"))
        if x == -1:
            self.s.sendall(bytes("SWAP\n", "utf-8"))
        else:
            self.s.sendall(bytes(f"{x},{y}\n", "utf-8"))
            self.board[x][y] = self.COLOUR_MAP[self.colour]
            self.skynet.make_move((x, y))


if __name__ == "__main__":
    agent = SkynetAgent()
    agent.run()
