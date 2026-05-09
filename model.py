import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_size, hiddens, output_size, device=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dim_list = [input_size, *hiddens, output_size]
        self.device = device

        # ---------------- FIXED BLOCK ---------------- #
        self.layers = []

        for dim1, dim2 in zip(self.dim_list[:-1], self.dim_list[1:]):
            self.layers.append(nn.Linear(dim1, dim2))

            # No activation after last layer
            if dim2 != self.dim_list[-1]:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.3))

        self.models = nn.Sequential(*self.layers)

        if self.device is not None:
            self.models = self.models.to(self.device)

    # ---------------- Forward ---------------- #
    def forward(self, input):

        assert input.shape[1] == self.input_size

        if self.device is not None:
            input = input.to(self.device)

        output = self.models(input)
        return output

    # ---------------- Move to CPU ---------------- #
    def to_cpu(self):
        self.device = None
        self.models = self.models.cpu()

    # ---------------- Move to GPU ---------------- #
    def to_cuda(self, device):
        self.device = device
        self.models = self.models.to(device)