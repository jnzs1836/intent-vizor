import torch
import torch.nn as nn


class TransposeModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, inputs):
        return inputs.transpose(1, 2)
