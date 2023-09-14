from Measures import EpochMeasure
from Networks import MLP

if __name__ == '__main__':
  args = {
    "learning_rate" : 0.01,
    "max_epochs": 10000,
    "to_train": ["insert-3", "insert-n", "reverse-n", "swap-2", "swap-3", "swap-n"],
    "min_n": 4,
    "max_n": 7,
    "network_label": "MLP-3-300",
    "results_path": "epoch_results.csv",
  }

  network_args = {
    "layers": [300, 300, 300]    
  }

  measure = EpochMeasure(MLP, args, network_args)
  measure.run_all()
  measure.save_results()
  measure.graph_results()