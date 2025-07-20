import os
import random

import numpy as np


def set_seed(seed: int):
    """
    全ての乱数シードを固定するための関数。
    実験の再現性を確保する。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
