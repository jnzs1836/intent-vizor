import os
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.metric import Metric
from evaluation import EvaluationMetric


class DebugLogger:
    def __init__(self, log_dir, save_dir, config, saving_criteria="f_measure_by_avg"):
        pass

    def log_step(self, step, losses, probs, metrics, step_type="train"):
        pass

    def log_train_step(self, step, scores, losses, probs, metrics):
        pass

    def log_valid_step(self, step, scores, losses, probs, metrics):
        pass

    def log_epoch(self, epoch_i, step_type):
        pass

    def log_train_epoch(self, epoch_i):
        pass

    def log_valid_epoch(self, epoch_i, model):
        pass

    def save_checkpoint(self, epoch_i, model):
        pass

    def _reset_epoch(self):
        pass
