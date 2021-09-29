import torch
from evaluation import EvaluationMetric


class SolverBase:
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0"

    def build(self):
        raise NotImplementedError("Solver method build is not implemented!")

    def evaluate_results(self, scores, losses, probs, batch):
        metrics = EvaluationMetric.generate_metrics(self.evaluation_methods, scores, losses, probs, batch)
        return metrics

    def step(self, batch_i, batch, step_type="train"):
        raise NotImplementedError("Solver method step is not implemented!")

    def train_step(self, batch_i, batch):
        self.model.train()
        return self.step(batch_i, batch, "train")

    def valid_step(self, batch_i, batch):
        self.model.eval()
        with torch.no_grad():
            return self.step(batch_i, batch, "valid")

    def get_model(self):
        return self.model
