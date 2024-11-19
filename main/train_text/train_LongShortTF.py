import torch
from torch.utils.data import Dataset
from transformers import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score