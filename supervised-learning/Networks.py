import torch
import torch.nn as nn
    
class MLP(nn.Module):
  def __init__(self, n, layers):
    super().__init__()

    layers = [n]+layers+[n]
    linear_modules = [nn.Linear(x,y) for x,y in zip(layers[:-1], layers[1:])]
    self.layers = nn.ModuleList(linear_modules)
    self.activation = nn.LeakyReLU()
  
  def forward(self, x):
    for layer in self.layers:
      x = self.activation(layer(x))
    return x

class Transformer(nn.Module):
  def __init__(self, n, layers):
    pass

  def forward(self, x):
    pass

