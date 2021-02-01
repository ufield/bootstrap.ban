import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class FCLayer(nn.Module):
    def __init__(self,
            input_dim,
            output_dim,
            activation='ReLU',
            dropout=0
        ):
        super(FCLayer, self).__init__()
        layers = []
        layers.append(nn.Dropout(dropout))
        if activation != '':
            layers.append(getattr(nn, activation)())
        layers.append(weight_norm(nn.Linear(input_dim, output_dim), dim=None))
        self.sequential = nn.Sequential(*layers)

        # self.dropout    = nn.Dropout(dropout)
        # if activation != '':
        #     self.activation = getattr(nn, activation)()
        #     # layers.append(getattr(nn, activation)())

        # # self.linear     = nn.Linear(input_dim, output_dim)
        # self.linear     = weight_norm(nn.Linear(input_dim, output_dim), dim=None)

    def forward(self, x):
        # x = self.linear(x)
        # out = self.dropout(x)
        # # if

        # if self.activation != '':
        #     out = self.activation(out)
        # return out
        return self.sequential(x)


if __name__ == '__main__':
    fc = FCLayer(10, 20, dropout=0.2)
    # fc.eval()
    x = torch.rand(5, 10)
    y = fc(x)

    print(x)
    print(y, y.shape)


    fc = FCLayer(50, 7, '', dropout=0)
    # fc.eval()
    x = torch.rand(5, 50)
    y = fc(x)

    print(x)
    print(y, y.shape)

