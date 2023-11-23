import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from converter import get_permutation_from_index

class PredictNextState(Dataset):
  def __init__(self, file_path, permutation_size):
    self.file_path = file_path
    self.permutation_size = permutation_size

    self.df = pd.read_csv(file_path)
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    x = get_permutation_from_index(self.df.iloc[index].values[0], self.permutation_size).astype(np.float32)
    y = get_permutation_from_index(self.df.iloc[index].values[1], self.permutation_size).astype(np.float32)
    
    return x, y
  

class PredictNextMove(Dataset):
  def __init__(self, file_path, perm_size):
    self.df = pd.read_csv(file_path)
    self.perm_size = perm_size

  def __getitem__(self, index):
    x = get_permutation_from_index(self.df.iloc[index]['state'], self.perm_size).astype(np.float32)
    i, j = [int(x) for x in self.df.iloc[index]['move'].split('-')]
    return x, i*self.perm_size+j
  
  def __len__(self):
    return len(self.df)