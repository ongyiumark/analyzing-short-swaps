from Measures import Measure
from Networks import MLP
from Datasets import PredictNextState

if __name__ == '__main__':
  args = {
    "learning_rate" : 0.01,
    "max_epochs": 10000,
    "to_train": ["insert-n", "swap-2", "swap-n"],
    "min_n": 4,
    "max_n": 7,
    "network_label": "MLP-3-300",
    "results_path": "epoch_results.csv",
    "device": "cuda:0",
    "batch_size": None
  }

  network_args = {
    "layers": [300, 300, 300]    
  }

  measure = Measure(MLP, PredictNextState, args, network_args)
  measure.run_all()
  measure.save_results()
  measure.graph_results()