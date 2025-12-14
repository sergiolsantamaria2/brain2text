import os
import random
import numpy as np
import torch

def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed is None or seed < 0:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
