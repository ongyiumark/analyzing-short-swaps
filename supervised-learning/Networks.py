import torch
import torch.nn as nn
import torch.nn.functional as F
    
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


class MLPMove(nn.Module):
  def __init__(self, n, layers):
    super().__init__()

    layers = [n]+layers
    linear_modules = [nn.Linear(x,y) for x,y in zip(layers[:-1], layers[1:])]
    self.layers = nn.ModuleList(linear_modules)
    self.activation = nn.LeakyReLU()
    self.out_layer = nn.Linear(layers[-1], n*n)
  
  def forward(self, x):
    for layer in self.layers:
      x = self.activation(layer(x))
    x = self.out_layer(x)
    return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class AttentionNetwork(nn.Module):
  def __init__(self, n, layers):
    super().__init__()
    self.n = n
    layers = [n]+layers+[n]
    linear_modules = [nn.Linear(x,y) for x,y in zip(layers[:-1], layers[1:])] 
    attention_weights = [nn.Linear(x,y) for x,y in zip(layers[:-1], layers[1:])] 
    self.layers = nn.ModuleList(linear_modules)
    self.attentions = nn.ModuleList(attention_weights)
    self.activation = nn.LeakyReLU()

  def forward(self, x):
    batch_size = x.shape[0]
    for attention, layer in zip(self.layers, self.attentions):
       pass
    return x