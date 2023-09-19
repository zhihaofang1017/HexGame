import torch
import torch.nn as nn
from utils import device, BaseModel


class PolicyModel(BaseModel):
    """
    Policy based move selection.

    It always assumes that the current player is the black/red player (with value 1 on board) who aims to connect top and bottom sides.

    Input: b*11*11 tensor
    Output: b*122 tensor
    """

    def __init__(self):
        super().__init__()
        """
        Input: b*27*12*12 tensor
        Output: b*122 tensor
        """
        self.pipeline = nn.Sequential(
            nn.Conv2d(27, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128, 168),
            nn.Sigmoid(),
            nn.Linear(168, 122)
        )

    def forward(self, x):
        """
        Input: b*11*11 tensor
        Output: b*122 tensor
        """
        x = self.encodeBoard(x)
        x = self.pipeline(x)
        return x


if __name__ == "__main__":
    model = PolicyModel().to(device)
    x = torch.zeros(1, 11, 11).to(device)
    output = model(x)
    print("output : \n{}".format(output))
    v = output[0].argmax()
    print("with the largest prob: {}".format(v))
    print(len(output[0]))
