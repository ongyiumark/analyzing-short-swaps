import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import subprocess as sub

p = [1, 2, 3]
command = ["./get-min", "3" ,"1" ,"2", "3"]
run_process = sub.run(command, stdout=sub.PIPE, stderr=sub.PIPE)
output = run_process.stdout.decode()
print(output)