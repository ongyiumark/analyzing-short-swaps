from Measures import Measure, RunUntilEpoch, PredictMove
from Networks import MLP, AttentionNetwork, MLPMove
from Datasets import PredictNextState, PredictNextMove

if __name__ == '__main__':
  args = {
    "learning_rate" : 0.01,
    "max_epochs": 10000,
    "to_train": ["swap-2", "swap-3", "swap-4", "swap-n", "reverse-4", "reverse-n"],
    "min_n": 4,
    "max_n": 8,
    "experiment_label": "Epoch_MLPMove-3-300_Batch-ALL",
    "results_path": "results-move.csv",
    "device": "cuda:0",
    "batch_size": None
  }

  network_args = {
    "layers": [300, 300, 300]    
  }

  measure = PredictMove(MLPMove, PredictNextMove, args, network_args)
  measure.run_all()
  measure.save_results()
  measure.graph_results()
  # args = {
  #   "learning_rate" : 0.01,
  #   "num_epochs": 200,
  #   "to_train": ["swap-2", "swap-3", "swap-4", "swap-n", "reverse-n", "reverse-4"],
  #   "perm_size": 5,
  #   "experiment_label": "Accuracy_MLP-3-300_Perm-5_Batch-ALL",
  #   "results_path": "results.csv",
  #   "device": "cuda:0",
  #   "batch_size": None
  # }

  # network_args = {
  #   "layers": [300, 300, 300]    
  # }

  # measure = RunUntilEpoch(MLP, PredictNextState, args, network_args)
  # # measure.run_all()
  # # measure.save_results()
  # measure.graph_results()