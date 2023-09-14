# IMPORT LIBRARIES
import torch
import torch.nn as nn
    
class MLP(nn.Module):
  def __init__(self, n, layers):
    super().__init__()
    layers = [n]+layers+[n]
    linear_modules = [nn.Linear(x,y) for x,y in zip(layers[:-1], layers[1:])]
    self.layers = nn.ModuleList(linear_modules)
  
  def forward(self, x):
    activation = nn.LeakyReLU()
    for layer in self.layers:
      x = activation(layer(x))
    
    return x

class ResNet(nn.Module):
  def __init__(self, n, layers):
    pass

  def forward(self, x):
    pass

