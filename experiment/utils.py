import numpy as np
import pandas as pd

import os
import random
import torch

from sklearn.metrics import cohen_kappa_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

def cal_kappa_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    kappa = cohen_kappa_score(data[label_col],pred,weights='quadratic')
    return kappa

