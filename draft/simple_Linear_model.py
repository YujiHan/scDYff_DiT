import torch
import torch.nn as nn
import numpy as np


class simple_linear(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
        )

        self.time_emb = nn.Linear(input_size, hidden_size, bias=True)
        self.time_emb_2 = nn.Linear(input_size, hidden_size * 2, bias=True)

        self.final = nn.Linear(hidden_size * 2, output_size, bias=True)

    def forward(self, x1, x3, t1, t2, t3):
        x1 = self.input_layer(x1)
        x3 = self.input_layer(x3)

        t1 = self.time_emb(t1)
        t2 = self.time_emb_2(t2)
        t3 = self.time_emb(t3)

        res1 = x1 * (1 + t1) + t1
        res3 = x3 * (1 + t3) + t3

        combined = torch.cat((res1, res3), dim=1)
        res2 = combined * (1 + t2) + t2

        output = self.final(res2)

        return output
