import os
import random
import numpy as np
import torch

def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cal_total_model_param(model_list):
    total = 0.
    for model in model_list:
        model_para = sum([param.nelement() for param in model.parameters()])
        total += model_para

    print("Number of parameter: %.2fM" % (total / 1e6))