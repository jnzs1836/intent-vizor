import torch
import torch.nn as nn


class ShotRandomGuess(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.score_net = nn.Linear(1, 1)
        self.topic_net = nn.Linear(1, 1)
        self.query_decoder = nn.Linear(1, 1)

    def forward(self, batch, shot_query):
        overall_scores = torch.rand(batch.size(0), batch.size(1))
        return overall_scores, {
            "topic_probs": None,
            "all_scores": None,
            "prior_loss":0
        }