"""

1. **Normalizer 类**：
   - `__init__`：在初始化过程中，将平均值和标准差初始化为零和一。这样做是为了确保如果用户没有调用 `fit` 方法进行标准化时，仍然可以进行零均值和单位方差的标准化。
   - `fit(tensor, dim=0, keepdim=False)`：这个方法用于计算给定张量的平均值和标准差。默认情况下，它计算每列的平均值和标准差。如果指定了 `dim` 参数，则沿指定维度计算。计算完成后，平均值和标准差将存储在实例的属性中。
   - `norm(tensor)`：这个方法用于对给定的张量进行标准化。它使用之前计算得到的平均值和标准差，将每个元素减去平均值，然后除以标准差。
   - `denorm(normed_tensor)`：对标准化后的张量进行反标准化，即乘以标准差然后加上平均值。
   - `state_dict()`：返回一个包含平均值和标准差的字典，用于保存当前状态。
   - `load_state_dict(state_dict)`：从给定的状态字典中加载平均值和标准差。
   - `from_state_dict(cls, state_dict)`：从状态字典中创建一个新的类实例，使用其中的平均值和标准差。

2. **Featurizer 类**：
   - `__init__`：在初始化过程中，接受一个允许的类型列表作为参数，并将其存储在实例中。
   - `get_fea(key)`：返回给定键的特征。这里假定 `_embedding` 字典中已经包含了特征，如果不存在则会引发断言错误。
   - `load_state_dict(state_dict)`：从给定的状态字典中加载特征到 `_embedding` 字典中，并更新允许的类型列表。
   - `get_state_dict()`：返回一个包含特征的字典。
   - `embedding_size` 属性：返回特征的维度大小，这里通过获取 `_embedding` 字典中第一个值的长度来确定。
   - `from_json(cls, embedding_file)`：从 JSON 文件加载特征，并返回一个类实例。这里假设 JSON 文件的格式是键值对，键是类型，值是特征。

3. **save_checkpoint 函数**：
   - 这个函数用于保存模型检查点。它将状态字典保存到文件中，并在需要时将最佳模型保存到另一个文件中。
"""


import gc
import json
import shutil
from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from tqdm.autonotebook import tqdm

# These functions are adpated from the implementation from roost by Rhys E. A. Goodall & Alpha A. Lee
# Source: https://github.com/CompRhys/roost

class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @classmethod
    def from_state_dict(cls, state_dict):
        instance = cls()
        instance.mean = state_dict["mean"].cpu()
        instance.std = state_dict["std"].cpu()

        return instance


class Featurizer:
    """Base class for featurizing nodes and edges."""

    def __init__(self, allowed_types):
        self.allowed_types = allowed_types
        self._embedding = {}
    #


    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = self._embedding.keys()

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(list(self._embedding.values())[0])

    @classmethod
    def from_json(cls, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = embedding.keys()
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance


def save_checkpoint(state, is_best, model_name, run_id):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """
    checkpoint = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
    best = f"models/{model_name}/best-r{run_id}.pth.tar"

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)
