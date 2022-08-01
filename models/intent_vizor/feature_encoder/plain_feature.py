import torch
import torch.nn as nn


class PlainFeatureEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, batch, frame_features,seg_len=None,concept1=None, concept2=None):
        return frame_features, frame_features


class PlainFusion(nn.Module):
    def __init__(self, use_slow_branch=False, use_fast_branch=True):
        nn.Module.__init__(self)
        self.use_slow_branch = use_slow_branch
        self.use_fast_branch = use_fast_branch

    def forward(self, frame_features, slow_result, fast_result):
        if self.use_fast_branch and self.use_slow_branch:
            return torch.cat([slow_result, fast_result], dim=2)
        elif self.use_slow_branch:
            return slow_result
        elif self.use_fast_branch:
            return fast_result

