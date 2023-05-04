import numpy as np
import random
import torch

seed = 42

def set_seed():
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'seed set to [{seed}]')