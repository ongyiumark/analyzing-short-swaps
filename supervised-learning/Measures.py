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
    tqdm_loader = tqdm(data_loader, leave=True)

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
    tqdm_loader = tqdm(data_loader, leave=True)

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
    
    if not os.path.isfile(self.args["results_path"]):
      df.to_csv(self.args["results_path"], index=False)

    current_df = pd.read_csv(self.args["results_path"])
    to_extend = []
    for i, row in df.iterrows():
      query_res = current_df.query(f"strategy==\"{row['strategy']}\" and size=={row['size']} and network==\"{row['network']}\"")
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
    fig, axes = plt.subplots(2, 3, figsize=(25,15))
    df = pd.read_csv(self.args["results_path"])
    df = df.query(f"network==\"{self.args['experiment_label']}\"")

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

    fig.suptitle(f"Training with {self.args['experiment_label']} until ACC $>0.95$.")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    fig.savefig(f"Epoch_{self.args['experiment_label']}_Results.png")
