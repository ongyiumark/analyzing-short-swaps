# IMPORT LIBRARIES
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class Measure:
  def __init__(self, ModelClass, DatasetClass, args, network_args):
    self.ModelClass = ModelClass
    self.DatasetClass = DatasetClass

    self.args = args
    self.network_args = network_args
    self.results = []

  def train_epoch(self, data_loader, model, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0.0
    count = 1
    tqdm_loader = tqdm(data_loader, leave=False)

    for batch_idx, (x,y) in enumerate(tqdm_loader):
      predictions = model(x)
      loss = loss_fn(predictions, y)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      tqdm_loader.set_postfix(average_loss=total_loss/count, epoch=epoch)
      total_loss += loss.item()
      count += 1

    mse = total_loss/count
    return mse

  def train(self, data_loader, model, n, strategy):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=self.args["learning_rate"])

    mse = float("inf")
    accuracy = 0
    epoch = 0
    while accuracy < 0.95 and epoch < self.args["max_epochs"]:
      epoch += 1
      mse = self.train_epoch(data_loader, model, optimizer, loss_fn, epoch)
      accuracy = self.test(data_loader, model)
    
      if epoch % 100 == 0:
        print(f"Strategy: {strategy}, size: {n}, epoch: {epoch}, MSE: {mse:.4f}, accuracy: {accuracy:.4f}")

    return epoch, mse
  
  def test(self, data_loader, model):
    model.eval()
    tqdm_loader = tqdm(data_loader, leave=False)

    correct = 0
    count = 0
    for batch_idx, (x,y) in enumerate(tqdm_loader):
      predictions = model(x)
      y = y.cpu().numpy().astype(int)
      predictions = (predictions.detach().cpu().numpy()+0.5).astype(int)

      for tx, ty in zip(predictions, y):
        correct += all(tx == ty)
        count += 1
      tqdm_loader.set_postfix(accuracy=correct/count)
    
    accuracy = correct/count
    return accuracy

  def run_one(self, n, strategy):
    file_path = f"./data/{strategy}/perm{n}.csv"
    dataset = self.DatasetClass(file_path, n, self.args["device"])

    model = self.ModelClass(n, **self.network_args).to(self.args["device"])
    batch_size = len(dataset) if self.args["batch_size"] is None else self.args["batch_size"]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epoch, mse = self.train(data_loader, model, n, strategy)
    accuracy = self.test(data_loader, model)

    return epoch, mse, accuracy
  
  def run_all(self):
    for strategy in self.args["to_train"]:
      tmp_results = []
      for n in range(self.args["min_n"], self.args["max_n"]+1):
        epoch, mse, accuracy = self.run_one(n, strategy)
        tmp_results.append((n, epoch, mse, accuracy))
        print(f"{strategy} - Size: {n}, epochs: {epoch},  MSE: {mse:.4f}, accuracy: {accuracy:.4f}")
      
      self.results.append(tmp_results)

  def save_results(self):
    to_df = []
    for strategy, result in zip (self.args["to_train"], self.results):
      for size, epoch, mse, acc in result:
        to_df.append([strategy, size, epoch, mse, acc, self.args["network_label"]])
    df = pd.DataFrame(to_df, columns=["strategy", "size", "epochs", "mse", "accuracy", "network"])
    
    if not os.path.isfile(self.args["results_path"]):
      df.to_csv(self.args["results_path"], index=False)

    current_df = pd.read_csv(self.args["results_path"])
    to_extend = []
    for i, row in df.iterrows():
      query_res = current_df.query(f"strategy==\"{row['strategy']}\" and size=={row['size']} and network==\"{row['network']}\"")
      if len(query_res) > 0:
        current_df.iloc[query_res.index[0]] = row
      else:
        to_extend.append(row)

    extended_df = pd.concat([current_df, pd.DataFrame(to_extend)], ignore_index=True)
    extended_df.to_csv(self.args["results_path"], index=False)

  def graph_results(self):
    fig, axes = plt.subplots(3, 4, figsize=(25,15))
    df = pd.read_csv(self.args["results_path"])
    df = df.query(f"network==\"{self.args['network_label']}\"")

    strategies = df["strategy"].unique()
    sizes = df["size"].unique()

    for i,ax in enumerate(axes.flatten()[:len(strategies)]):
      curr_df = df.query(f"strategy==\"{strategies[i]}\"")

      ax.plot(range(min(sizes), max(sizes)+1), curr_df["epochs"].values, label="Epoch")
      
      acc_ax = ax.twinx()
      acc_ax.plot(range(min(sizes), max(sizes)+1), curr_df["accuracy"].values, label="Accuracy", color="orange")
      acc_ax.set_ylim(0.9,1)

      ax.set_ylim(0, self.args["max_epochs"])
      ax.set_xticks(range(min(sizes), max(sizes)+1))
      
      ax.set_title(strategies[i])
      ax.set_xlabel("Permutation Size")
      ax.set_ylabel("Epochs")

      ax.legend(loc="upper left")
      acc_ax.legend(loc="upper right")

    fig.suptitle(f"Training with {self.args['network_label']} until ACC $>0.95$.")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    fig.savefig(f"Epoch_{self.args['network_label']}_Results.png")


# class EpochMeasure:
#   def __init__(self, NetworkTemplate, args, network_args):
#     self.NetworkTemplate = NetworkTemplate
#     self.args = args
#     self.network_args = network_args
#     self.results = []
  
#   def train(self, model, dataX, dataY, n, strategy):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=self.args["learning_rate"])

#     mse = float('inf')
#     epoch = 0
#     while mse > 0.01 and epoch < self.args["max_epochs"]:
#       outputs = model(dataX)
#       optimizer.zero_grad()
#       loss = criterion(outputs, dataY)
#       loss.backward()
#       optimizer.step()

#       if epoch % 100 == 0:
#         print(f"Strategy: {strategy}, size: {n}, epoch: {epoch}, loss: {loss.item():1.5f}")
      
#       epoch += 1
#       mse = loss.item()
    
#     return epoch, mse
    
#   def test(self, model, dataX, dataY):
#     model.eval()
#     predictions = model(dataX)
#     dataY = dataY.numpy().astype(int)
#     predictions = (predictions.detach().numpy()+0.5).astype(int)

#     accuracy = 0
#     for x,y in zip(predictions, dataY):
#       accuracy += all(x == y)
#     accuracy /= len(dataX)

#     return accuracy

#   def run_one(self, n, directory):
#     df = pd.read_csv(f"../data/{directory}/perm{n}.csv")
#     dataX = df['state'].values
#     dataY = df['next_state'].values

#     n_dataX = []
#     n_dataY = []
#     for i, (x, y) in enumerate(zip(dataX, dataY)):
#       n_dataX.append(get_permutation_of_index(x, n))
#       n_dataY.append(get_permutation_of_index(y, n))
#       if i % 100 == 0:
#         print(f"Converting for size {n}: {i}/{len(dataX)}")

#     n_dataX = torch.tensor(np.array(n_dataX), dtype=torch.float32)
#     n_dataY = torch.tensor(np.array(n_dataY), dtype=torch.float32)

#     model = self.NetworkTemplate(n, **self.network_args)
#     epoch, mse = self.train(model, n_dataX, n_dataY, n, directory)
#     accuracy = self.test(model, n_dataX, n_dataY)

#     return epoch, mse, accuracy

#   def run_all(self):
#     for directory in self.args["to_train"]:
#       tmp_results = []
#       for n in range(self.args["min_n"], self.args["max_n"]+1):
#         epoch, mse, accuracy = self.run_one(n, directory)
#         tmp_results.append((n, epoch, mse, accuracy))
#         print(f"{directory} - Size: {n}, epochs: {epoch},  MSE: {mse:.4f}, accuracy: {accuracy:.4f}")
      
#       self.results.append(tmp_results)
    
#   def save_results(self):
#     to_df = []
#     for strategy, result in zip (self.args["to_train"], self.results):
#       for size, epoch, mse, acc in result:
#         to_df.append([strategy, size, epoch, mse, acc, self.args["network_label"]])
#     df = pd.DataFrame(to_df, columns=["strategy", "size", "epochs", "mse", "accuracy", "network"])
    
#     if not os.path.isfile(self.args["results_path"]):
#       df.to_csv(self.args["results_path"], index=False)

#     current_df = pd.read_csv(self.args["results_path"])
#     to_extend = []
#     for i, row in df.iterrows():
#       query_res = current_df.query(f"strategy==\"{row['strategy']}\" and size=={row['size']} and network==\"{row['network']}\"")
#       if len(query_res) > 0:
#         current_df.iloc[query_res.index[0]] = row
#       else:
#         to_extend.append(row)

#     extended_df = pd.concat([current_df, pd.DataFrame(to_extend)], ignore_index=True)
#     extended_df.to_csv(self.args["results_path"], index=False)

#   def graph_results(self):
#     fig, axes = plt.subplots(2, 3, figsize=(20,10))
#     df = pd.read_csv(self.args["results_path"])
#     df = df.query(f"network==\"{self.args['network_label']}\"")

#     strategies = df["strategy"].unique()
#     sizes = df["size"].unique()

#     for i,ax in enumerate(axes.flatten()[:len(strategies)]):
#       curr_df = df.query(f"strategy==\"{strategies[i]}\"")

#       ax.plot(range(min(sizes), max(sizes)+1), curr_df["epochs"].values, label="Epoch")
      
#       acc_ax = ax.twinx()
#       acc_ax.plot(range(min(sizes), max(sizes)+1), curr_df["accuracy"].values, label="Accuracy", color="orange")
#       acc_ax.set_ylim(0.9,1)

#       ax.set_ylim(0, self.args["max_epochs"])
#       ax.set_xticks(range(min(sizes), max(sizes)+1))
      
#       ax.set_title(strategies[i])
#       ax.set_xlabel("Permutation Size")
#       ax.set_ylabel("Epochs")

#       ax.legend(loc="upper left")
#       acc_ax.legend(loc="upper right")

#     fig.suptitle(f"Training with {self.args['network_label']} until MSE $<0.01$.")
#     fig.tight_layout()

#     fig.savefig(f"Epoch_{self.args['network_label']}_Results.png")

