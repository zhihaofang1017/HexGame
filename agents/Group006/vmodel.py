import torch
import torch.nn as nn
from utils import device, BaseModel


class ValueModel(BaseModel):
    """
    CNN function evaluator for Skynet.

    It always assumes that the current player is the black/red player (with value 1 on board) who aims to connect top and bottom sides.

    Input: b*11*11 tensor
    Output: b*1 tensor
    """
    CONV_CHANNELS = 128
    LINEAR_CHANNELS = 128

    def __init__(self):
        super().__init__()
        """
        Input: b*27*12*12 tensor
        Output: b*1 tensor
        """
        self.pipeline = nn.Sequential(
            nn.Conv2d(27, self.CONV_CHANNELS, 3),
            nn.BatchNorm2d(self.CONV_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(self.CONV_CHANNELS, self.CONV_CHANNELS, 3),
            nn.BatchNorm2d(self.CONV_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(self.CONV_CHANNELS, self.CONV_CHANNELS, 3),
            nn.BatchNorm2d(self.CONV_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(self.CONV_CHANNELS, self.CONV_CHANNELS, 3),
            nn.BatchNorm2d(self.CONV_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(self.CONV_CHANNELS, self.CONV_CHANNELS, 3),
            nn.BatchNorm2d(self.CONV_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(self.CONV_CHANNELS, self.CONV_CHANNELS, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.CONV_CHANNELS, self.LINEAR_CHANNELS),
            nn.Sigmoid(),
            nn.Linear(self.LINEAR_CHANNELS, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Input: b*11*11 tensor
        Output: b*1 tensor
        """
        x = self.encodeBoard(x)
        x = self.pipeline(x)
        return x


def hack_weights():
    # torch.set_printoptions(threshold=10000)

    PRETRAINED_PATH = "weights/vmodel/1130-8-ep6250_gb64_loss13.0380.pth"
    SAVE_PATH = "weights/vmodel/1130-8-ep6250_gb64_loss13.0380-64to128.pth"

    PERTURBATION = 0.01

    old_state_dict = torch.load(PRETRAINED_PATH, map_location=device)
    print(len(old_state_dict))
    print(old_state_dict.keys())

    evaluator = ValueModel().to(device)
    print(evaluator)
    with torch.no_grad():
        for layer, param in evaluator.state_dict().items():
            _, layer_i, layer_type = layer.split(".")
            layer_i = int(layer_i)
            if layer_i not in [0, 3, 6, 9, 12, 15, 18, 20]:
                continue
            # print(layer_i, layer_type)
            # print(param.shape)
            old_weight = old_state_dict[layer]
            old_weight_2 = old_weight.clone()
            old_weight_2 += torch.randn_like(old_weight_2) * PERTURBATION
            old_weight_3 = old_weight.clone()
            old_weight_3 += torch.randn_like(old_weight_3) * PERTURBATION
            old_weight_4 = old_weight.clone()
            old_weight_4 += torch.randn_like(old_weight_4) * PERTURBATION
            if layer_i == 0:
                if layer_type == 'weight':
                    new_weight = torch.cat((old_weight, old_weight_2), dim=0)
                elif layer_type == 'bias':
                    new_weight = torch.cat((old_weight, old_weight_2), dim=0)
            elif layer_i in (3, 6, 9, 12, 15, 18):
                if layer_type == 'weight':
                    new_weight_1 = torch.cat((old_weight, old_weight_2), dim=1)
                    new_weight_2 = torch.cat((old_weight_3, old_weight_4), dim=1)
                    new_weight = torch.cat((new_weight_1, new_weight_2), dim=0)
                elif layer_type == 'bias':
                    new_weight = torch.cat((old_weight, old_weight_2), dim=0)
                    new_weight = new_weight * 2
            elif layer_i == 20:
                if layer_type == 'weight':
                    new_weight = torch.cat((old_weight, old_weight_2), dim=1)
                elif layer_type == 'bias':
                    new_weight = old_weight * 2
            param.copy_(new_weight)

    torch.save(evaluator.state_dict(), SAVE_PATH)
