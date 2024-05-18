from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim,device):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.hidden_dim = hidden_dim
        self.intput_dim = input_dim
        self.input_length = input_length
        self.output_dim = output_dim
        self.Wx = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wy = nn.Linear(hidden_dim, output_dim)




    def forward(self, x):
        # Implementation here ...
        h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        for i in range(self.input_length):
            h = torch.tanh(self.Wx(x[:, i, :]) + self.Wh(h))
        y = self.Wy(h)
        return nn.functional.softmax(y, dim=1)
        
    # add more methods here if needed
