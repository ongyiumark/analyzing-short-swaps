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
      x = x.to(self.args["device"])
      y = y.to(self.args["device"])

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
      x = x.to(self.args["device"])
      y = y.to(self.args["device"])

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
    file_path = f"../data/{strategy}/perm{n}.csv"
    dataset = self.DatasetClass(file_path, n)

    model = self.ModelClass(n, **self.network_args).to(self.args["device"])
    batch_size = len(dataset) if self.args["batch_size"] is None else self.args["batch_size"]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_path = f"models/{self.args['experiment_label']}/{strategy}/perm{n}.pt"
    if os.path.exists(model_path):
      state_dict = torch.load(model_path)
      model.load_state_dict(state_dict['model'])
      epoch = state_dict['epoch']
      mse = state_dict['mse']
    else:
      epoch, mse = self.train(data_loader, model, n, strategy)
      state_dict = dict()
      state_dict['model'] = model.state_dict()
      state_dict['epoch'] = epoch
      state_dict['mse'] = mse

      os.makedirs(os.path.dirname(model_path), exist_ok=True)
      torch.save(state_dict, model_path)
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
        to_df.append([strategy, size, epoch, mse, acc, self.args["experiment_label"]])
    df = pd.DataFrame(to_df, columns=["strategy", "size", "epochs", "mse", "accuracy", "network"])
    
    new_df = False
    if not os.path.isfile(self.args["results_path"]):
      df.to_csv(self.args["results_path"], index=False)
      new_df = True
  
    current_df = pd.read_csv(self.args["results_path"])
    to_extend = []
    for i, row in df.iterrows():
      query_res = current_df.query(f"strategy==\"{row['strategy']}\" and size=={row['size']} and network==\"{row['network']}\"")
      if len(query_res) > 0:
        current_df.iloc[query_res.index[0]] = row
        if not new_df:
          print(f"Found an existing result for {row['strategy']} with size {row['size']} and {row['network']}. If you'd like to replace this result, remove the corresponding row from {self.args['results_path']} and run this program again.")
      else:
        to_extend.append(row)

    if len(to_extend) > 0 and len(current_df) > 0:
      extended_df = pd.concat([current_df, pd.DataFrame(to_extend)], ignore_index=True)
    elif len(current_df) > 0:
      extended_df = current_df
    else:
      extended_df = pd.DataFrame(to_extend)
    extended_df.to_csv(self.args["results_path"], index=False)

  def graph_results(self):
    fig, axes = plt.subplots(3, 2, figsize=(15,18))
    df = pd.read_csv(self.args["results_path"])
    df = df.query(f"network==\"{self.args['experiment_label']}\"")

    strategies = df["strategy"].unique()
    sizes = df["size"].unique()

    for i,ax in enumerate(axes.flatten()[:len(strategies)]):
      curr_df = df.query(f"strategy==\"{strategies[i]}\"")


      ax.scatter(range(min(sizes), max(sizes)+1), curr_df["epochs"].values, s=100, color="red")
      ax.plot(range(min(sizes), max(sizes)+1), curr_df["epochs"].values, label="Epoch", linewidth=2.5)
      
      # acc_ax = ax.twinx()
      # acc_ax.plot(range(min(sizes), max(sizes)+1), curr_df["accuracy"].values, label="Accuracy", color="orange")
      # acc_ax.set_ylim(0.9,1)

      ax.set_ylim(0, max(df["epochs"].values)*1.1)
      ax.set_xticks(ticks=range(min(sizes), max(sizes)+1), labels=range(min(sizes), max(sizes)+1), fontsize=12)
      ax.set_yticks(ticks=ax.get_yticks(), labels=ax.get_yticklabels(), fontsize=12)
      
      strat, bound = strategies[i].split('-')
      if strat == "swap":
        strat = "swaps"
      if strat == "reverse":
        strat = "reversals"
      title = f"{bound}-bounded {strat}"

      if bound == "n":
        title = f"unbounded {strat}"

      title = title.title()

      ax.set_title(title, fontsize=14)
      ax.set_xlabel("Permutation Size", fontsize=12)
      ax.set_ylabel("Epochs", fontsize=12)

      ax.legend(loc="upper left")
      # acc_ax.legend(loc="upper right")

    fig.suptitle(f"Training {self.args['experiment_label']} until Accuracy >0.95.", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)

    fig.savefig(f"{self.args['experiment_label']}_Results.png")

class RunUntilEpoch:
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
      x = x.to(self.args["device"])
      y = y.to(self.args["device"])

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

  def train(self, data_loader, model, strategy):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=self.args["learning_rate"])

    mse = float("inf")
    accuracy = 0
    epoch = 0
    tmp_results = []
    while epoch < self.args["num_epochs"]:
      epoch += 1

      model_path = f"models/{self.args['experiment_label']}/{strategy}/epoch{epoch}.pt"
      if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['model'])
        mse = state_dict['mse']
      else:
        mse = self.train_epoch(data_loader, model, optimizer, loss_fn, epoch)
        state_dict = dict()
        state_dict['model'] = model.state_dict()
        state_dict['mse'] = mse

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(state_dict, model_path)
      
      accuracy = self.test(data_loader, model)

      tmp_results.append((epoch, mse, accuracy))
      if epoch % 100 == 0:
        print(f"Strategy: {strategy}, size: {self.args['perm_size']}, epoch: {epoch}, MSE: {mse:.4f}, accuracy: {accuracy:.4f}")

    self.results.append(tmp_results)
    return mse
  
  def test(self, data_loader, model):
    model.eval()
    tqdm_loader = tqdm(data_loader, leave=False)

    correct = 0
    count = 0
    for batch_idx, (x,y) in enumerate(tqdm_loader):
      x = x.to(self.args["device"])
      y = y.to(self.args["device"])

      predictions = model(x)
      y = y.cpu().numpy().astype(int)
      predictions = (predictions.detach().cpu().numpy()+0.5).astype(int)

      for tx, ty in zip(predictions, y):
        correct += all(tx == ty)
        count += 1
      tqdm_loader.set_postfix(accuracy=correct/count)
    
    accuracy = correct/count
    return accuracy

  def run_one(self, strategy):
    n = self.args['perm_size']
    file_path = f"../data/{strategy}/perm{n}.csv"
    dataset = self.DatasetClass(file_path, n)

    model = self.ModelClass(n, **self.network_args).to(self.args["device"])
    batch_size = len(dataset) if self.args["batch_size"] is None else self.args["batch_size"]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mse = self.train(data_loader, model, strategy)
    accuracy = self.test(data_loader, model)

    return mse, accuracy
  
  def run_all(self):
    for strategy in self.args["to_train"]:
      self.run_one(strategy)

  def save_results(self):
    to_df = []
    for strategy, result in zip (self.args["to_train"], self.results):
      for epoch, mse, acc in result:
        to_df.append([strategy, self.args['perm_size'], epoch, mse, acc, self.args["experiment_label"]])
    df = pd.DataFrame(to_df, columns=["strategy", "size", "epochs", "mse", "accuracy", "network"])

    if not os.path.isfile(self.args["results_path"]):
      df.to_csv(self.args["results_path"], index=False)

    current_df = pd.read_csv(self.args["results_path"])
    to_extend = []
    for i, row in df.iterrows():
      query_res = current_df.query(f"strategy==\"{row['strategy']}\" and size=={row['size']} and network==\"{row['network']}\" and epochs=={row['epochs']}")
      if len(query_res) > 0:
        current_df.iloc[query_res.index[0]] = row
        print(f"Found an existing result for {row['strategy']} with size {row['size']} and {row['network']}. If you'd like to replace this result, remove the corresponding row from {self.args['results_path']} and run this program again.")
      else:
        to_extend.append(row)

    if len(to_extend) > 0 and len(current_df) > 0:
      extended_df = pd.concat([current_df, pd.DataFrame(to_extend)], ignore_index=True)
    elif len(current_df) > 0:
      extended_df = current_df
    else:
      extended_df = pd.DataFrame(to_extend)
    extended_df.to_csv(self.args["results_path"], index=False)

  def graph_results(self):
    fig, axes = plt.subplots(3, 2, figsize=(15,18))
    df = pd.read_csv(self.args["results_path"])
    df = df.query(f"network==\"{self.args['experiment_label']}\"")

    strategies = df["strategy"].unique()
    epochs = df["epochs"].unique()

    for i,ax in enumerate(axes.flatten()[:len(strategies)]):
      curr_df = df.query(f"strategy==\"{strategies[i]}\"")
      
      ax.plot(range(min(epochs), max(epochs)+1), curr_df["accuracy"].values, label="Accuracy", linewidth=2.5)
      ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), fontsize=12)
      ax.set_yticks(ticks=ax.get_yticks(), labels=ax.get_yticklabels(), fontsize=12)
      ax.set_ylim(0,1.1)
      ax.set_xlim(min(epochs), max(epochs)+1)

      strat, bound = strategies[i].split('-')
      if strat == "swap":
        strat = "swaps"
      if strat == "reverse":
        strat = "reversals"
      title = f"{bound}-bounded {strat}"

      if bound == "n":
        title = f"unbounded {strat}"

      title = title.title()
      
      ax.set_title(title, fontsize=14)
      ax.set_xlabel("Number of Epochs", fontsize=12)
      ax.set_ylabel("Accuracy", fontsize=12)

      ax.legend(loc="upper left")

    fig.suptitle(f"Accuracy of {self.args['experiment_label']} by Epochs.", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)

    fig.savefig(f"{self.args['experiment_label']}_Results.png")