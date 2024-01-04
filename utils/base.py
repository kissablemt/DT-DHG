import numpy as np
import random
import torch
from scipy.stats import boxcox, zscore

eps = np.finfo(np.float32).eps

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mat_cat_mat(a, b):
    """
    a: N×Da
    b: M×Db
    c: N×M×(Da+Db)
        where c[i, j] == cat(a[i], b[j])
    """
    N = a.shape[0]
    M = b.shape[0]
    Da = a.shape[1]
    Db = b.shape[1]
    
    ar = a.unsqueeze(1).repeat(1, M, 1)
    br = b.unsqueeze(0).repeat(N, 1, 1)
    c = torch.cat((ar, br), dim=2)
    c = c.view(N, M, -1)
    return c

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return round(obj.item(), 4)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.device):
        return str(obj)
    else:
        return obj

def degree_matrix(A):
    D = torch.diag(torch.sum(A, dim=1))
    return D


def degree(A):
    D = degree_matrix(A)
    degree = D.dot(torch.ones(D.shape[0]))
    return degree


def minmax_rescale(x, min=0, max=1):
    x = x - x.min()
    x = x / x.max()
    x = x * (max - min) + min
    return x

def median_rescale(x, mu=0.5):
    bias = np.log(mu) / torch.log(torch.median(x))
    return  torch.pow(x, bias)

def boxcox_rescale(x):
    res, _ = boxcox((x + np.finfo(np.float32).eps).ravel())
    return res.reshape(x.shape)

def zscore_rescale(x):
    return zscore(x)


def rank_vec(vec, p_list, v_list):
    """
    demo:
        vec = torch.arange(100).view(10, 10) + 1
        p_list = [  95,   90,   85,   80,   75]
        v_list = [1.00, 0.60, 0.30, 0.15, 0.10]
        new_vec = rank_vec(vec, p_list, v_list)
        print(new_vec)
    """
    device = vec.device
    p_list = np.percentile(vec.detach().cpu(), p_list)

    def _dfs(i):
        if i == len(p_list):
            return torch.zeros_like(vec)
        else:
            return torch.where(p_list[i] <= vec, v_list[i], _dfs(i+1))

    return _dfs(i=0).to(device)


def global_rank(x, percentile=True):
    device = x.device

   # 使用 argsort() 函数获取张量中每个元素的排名
    xf = x.flatten()
    ranked_indices = torch.argsort(-xf) 
    
    # 查找每个元素在排名数组中的位置
    ranks = torch.zeros_like(xf).to(device)
    ranks[ranked_indices] = torch.arange(x.numel(), dtype=x.dtype).to(device)
    if percentile:
        ranks = (ranks / (x.numel() - 1)) * 100
    return ranks.view(x.shape)

def local_rank(x, percentile=True):
    device = x.device
    # 使用 argsort() 函数获取张量中每个元素在每行的排名
    ranked_indices = torch.argsort(-x, dim=1)

    # 查找每个元素在排名数组中的位置
    ranks = torch.zeros_like(x).to(device)
    for i, row in enumerate(ranked_indices):
        ranks[i, row] = torch.arange(x.shape[1], dtype=x.dtype).to(device)
    if percentile:
        ranks = (ranks / (x.shape[1] - 1)) * 100
    return ranks