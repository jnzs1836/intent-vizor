import os
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.metric import Metric
from evaluation import EvaluationMetric


class Logger:
    def __init__(self, log_dir, save_dir, config, saving_criteria="f_measure_by_avg"):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir
        self.config = config
        self.criteria_values_epoch_valid = []
        self.criteria_values_epoch_train = []
        self.criteria_values_epoch_test = []
        self.saving_criteria = saving_criteria
        self.metrics = {
            "train": {

            },
            "valid": {

            },
            "test": {

            },
            "pretrain": {

            }
        }
        self.criteria_optimal = EvaluationMetric.get_evaluation_optimal(saving_criteria)
        self.best_state_dict = None
        self.best_epoch = -1

    def get_desc(self, step_type):
        if step_type == "train":
            desc = "Train"
        elif step_type == "valid":
            desc = "Valid"
        elif step_type == "pretrain":
            desc = "Pretrain"
        else:
            desc = "Test"
        return desc

    def log_step(self, step, losses, probs, metrics, step_type="train"):
        # desc = "Train" if step_type == "train" else "Valid"
        desc = self.get_desc(step_type)
        for loss_name in losses:
            self.writer.add_scalar(desc + "_" + "Loss/{}".format(loss_name), losses[loss_name], step)
        # self.writer.add_scalar(desc + "_" + "Loss/recon_loss", losses['recon_loss'], step)
        # self.writer.add_scalar(desc + "_" + "Loss/prior_loss", losses['prior_loss'], step)
        # self.writer.add_scalar(desc + "_" + "Loss/sparsity_loss", losses['sparsity_loss'], step)
        # self.writer.add_scalar(desc + "_" + "Loss/gan_loss", losses['gan_loss'], step)

        # self.writer.update_loss(s_e_loss.data, step, 's_e_loss')
        # self.writer.update_loss(d_loss.data, step, 'd_loss')
        # self.writer.update_loss(c_loss.data, step, 'c_loss')
        for prob_name in probs:
            self.writer.add_scalar(desc + "_" + "Prob/{}".format(prob_name), probs[prob_name], step)
        # self.writer.add_scalar(desc + "_" + "Prob/original_prob", probs['original_prob'], step)
        # self.writer.add_scalar(desc + "_" + "Prob/fake_prob", probs['fake_prob'], step)
        # self.writer.add_scalar(desc + "_" + "Prob/uniform_prob", probs['uniform_prob'], step)
        # self.writer.add_scalar(desc + "_" + "Loss/s_e_loss", losses['s_e_loss'], step)
        # self.writer.add_scalar(desc + "_" + "Loss/d_loss", losses['d_loss'], step)
        # self.writer.add_scalar(desc + "_" + "Loss/c_loss", losses['c_loss'], step)
        for metric_name in metrics:
            if metric_name not in self.metrics[step_type]:
                self.metrics[step_type][metric_name] = Metric(metric_name)
            self.metrics[step_type][metric_name].add_value(metrics[metric_name])
    def log_train_step(self, step, scores, losses, probs, metrics):
        return self.log_step(step, losses, probs, metrics, "train")

    def log_valid_step(self, step, scores, losses, probs, metrics):
        return self.log_step(step, losses, probs, metrics, "valid")

    def log_test_step(self, step, scores, losses, probs, metrics):
        return self.log_step(step, losses, probs, metrics, "test")

    def log_pretrain_step(self, step, scores, losses, probs, metrics):
        return self.log_step(step, losses, probs, metrics, "pretrain")

    def log_epoch(self, epoch_i, step_type):
        # desc = "Train" if step_type == "train" else "Valid"
        desc = self.get_desc(step_type)
        metrics = self.metrics[step_type]
        for metric_name in metrics:
            metric = metrics[metric_name]
            metric_avg = metric.avg()
            self.writer.add_scalar(desc + "_" + "Metrics/{}".format(metric_name), metric_avg, epoch_i)
        if step_type == "train":
            criteria_value = metrics[self.saving_criteria].avg()
            self.criteria_values_epoch_train.append(criteria_value)
        elif step_type == "valid":
            criteria_value = metrics[self.saving_criteria].avg()
            self.criteria_values_epoch_valid.append(criteria_value)
        elif step_type == "pretrain":
            pass
        else:
            criteria_value = metrics[self.saving_criteria].avg()
            self.criteria_values_epoch_test.append(criteria_value)
    def log_train_epoch(self, epoch_i):
        self.log_epoch(epoch_i, "train")

    def log_valid_epoch(self, epoch_i, model):
        self.log_epoch(epoch_i, "valid")
        self.save_checkpoint(epoch_i, model)
        # self._reset_epoch()

    def log_pretrain_epoch(self, epoch_i):
        self.log_epoch(epoch_i, "pretrain")

    def log_test_epoch(self, epoch_i, model):
        self.log_epoch(epoch_i, "test")
        # self.save_checkpoint(epoch_i, model)
        self._reset_epoch()

    def save_checkpoint(self, epoch_i, model):
        last_criteria_value = self.criteria_values_epoch_valid[-1]
        optimal_criteria_value = None
        if self.criteria_optimal == "max":
            optimal_criteria_value = max(self.criteria_values_epoch_valid)
        elif self.criteria_optimal == "min":
            optimal_criteria_value = min(self.criteria_values_epoch_valid)
        if last_criteria_value == optimal_criteria_value:
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_i
        else:
            pass
        ckpt_path = os.path.join(str(self.save_dir), 'epoch.pkl')
        backup_ckpt_path = os.path.join(str(self.save_dir), 'epoch-backup.pkl')
        checkpoint = {
            "args": self.config.__dict__,
            "epoch": epoch_i,
            "epoch_state_dict": model.state_dict(),
            "best_state_dict": self.best_state_dict,
            "criteria_name": self.saving_criteria,
            "criteria_values_epoch_valid": self.criteria_values_epoch_valid,
            "criteria_values_epoch_train": self.criteria_values_epoch_train,
            "criteria_values_epoch_test": self.criteria_values_epoch_test,
            "best_epoch": self.best_epoch
        }
        torch.save(checkpoint, ckpt_path)
        torch.save(checkpoint, backup_ckpt_path)
        if epoch_i % 10 ==0:
            epoch_ckpt_path = os.path.join(str(self.save_dir), 'epoch-{}.pkl'.format(epoch_i))
            torch.save(checkpoint, epoch_ckpt_path)
    def _reset_epoch(self):
        self.metrics = {
            "train": {},
            "valid": {},
            "test": {}
        }
