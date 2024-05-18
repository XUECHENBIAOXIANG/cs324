from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.input_layer = nn.Linear(n_inputs, n_hidden[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)])
        self.output_layer = nn.Linear(n_hidden[-1], n_classes)
        #nn.Sequential 就是一个 nn.Module 的子类，也就是 nn.Module 所有的方法 (method) 它都有。并且直接使用 nn.Sequential 不用写 forward 函数，因为它内部已经帮你写好了。如果你确定 nn.Sequential 里面的顺序是你想要的，而且不需要再添加一些其他处理的函数
    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """

        x = self.input_layer(x)
        x = nn.functional.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.functional.relu(x)

        out = self.output_layer(x)
        return out
