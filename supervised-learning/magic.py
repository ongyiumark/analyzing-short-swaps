import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from converter import get_index_from_permutation_python, get_permutation_from_index_python
from Networks import MLP

def train(model, dataX, dataY, n, strategy):
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  mse = float('inf')
  epoch = 0
  while mse > 0.01 and epoch < 10000:
    outputs = model(dataX)
    optimizer.zero_grad()
    loss = criterion(outputs, dataY)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f"Strategy: {strategy}, size: {n}, epoch: {epoch}, loss: {loss.item():1.5f}")
    
    epoch += 1
    mse = loss.item()
  
  return epoch, mse
  
def test(model, dataX, dataY):
  model.eval()
  predictions = model(dataX)
  dataY = dataY.numpy().astype(int)
  predictions = (predictions.detach().numpy()+0.5).astype(int)

  accuracy = 0
  for x,y in zip(predictions, dataY):
    accuracy += all(x == y)
  accuracy /= len(dataX)

  return accuracy
  
strategy = "swap-3"
n = 6
if __name__ == "__main__":
  df_hi = pd.read_csv(f"../data/{strategy}/perm{n}.csv")
  dataX = df_hi['state'].values
  dataY = df_hi['next_state'].values

  n_dataX = []
  n_dataY = []
  for i, (x, y) in enumerate(zip(dataX, dataY)):
    n_dataX.append(get_permutation_from_index_python(x, n))
    n_dataY.append(get_permutation_from_index_python(y, n))


  n_dataX = torch.tensor(np.array(n_dataX), dtype=torch.float32)
  n_dataY = torch.tensor(np.array(n_dataY), dtype=torch.float32)

  model = MLP(n, [300,300,300])
  epoch, mse = train(model, n_dataX, n_dataY, n, strategy)

  for j in range(n,3,-1):
    df = pd.read_csv(f"../data/{strategy}/perm{j}.csv")
    dataX = df['state'].values
    dataY = df['next_state'].values
    n_dataX = []
    n_dataY = []
    for i, (x, y) in enumerate(zip(dataX, dataY)):
      n_dataX.append(get_permutation_from_index_python(x, n))
      n_dataY.append(get_permutation_from_index_python(y, n))
    n_dataX = torch.tensor(np.array(n_dataX), dtype=torch.float32)
    n_dataY = torch.tensor(np.array(n_dataY), dtype=torch.float32)

    accuracy = test(model, n_dataX, n_dataY)
    print(f"Size {j} accuracy: {accuracy}")

    
