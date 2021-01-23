import torch
import torch.nn as nn
import torch.nn.functional as F

class SLP(nn.Module):

    def __init__(self,
            input_dim,
            output_dim,
            activation='softmax'):
        super(SLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        out = F.__dict__[self.activation](x)
        return out
