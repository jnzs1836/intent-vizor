import torch
import torch.nn as nn


class UpSampling(nn.Module):
    def __init__(self, slow_feature_dim, fast_feature_dim, fusion_dim):
        nn.Module.__init__(self)
        self.slow_upsample = nn.Upsample(scale_factor=(1, 16), mode="bicubic", align_corners=False)
        self.fast_upsample = nn.Upsample(scale_factor=(1, 4), mode="bicubic", align_corners=False)
        self.slow_mlp = nn.Sequential(
            nn.Linear(slow_feature_dim, slow_feature_dim),
            nn.ReLU(),
            nn.Linear(slow_feature_dim, fusion_dim)
        )
        self.fast_mlp = nn.Sequential(
            nn.Linear(fast_feature_dim, fast_feature_dim),
            nn.ReLU(),
            nn.Linear(fast_feature_dim, fusion_dim)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.slow_mlp[0].weight.data)
        nn.init.kaiming_normal_(self.slow_mlp[2].weight.data)
        nn.init.kaiming_normal_(self.fast_mlp[0].weight.data)
        nn.init.kaiming_normal_(self.fast_mlp[2].weight.data)

    def forward(self, frame_features, slow_result, fast_result):
        slow_result = self.slow_upsample(slow_result.transpose(1, 2).unsqueeze(2))
        slow_result = slow_result.squeeze(2).transpose(1, 2)
        slow_result = self.slow_mlp(slow_result)
        fast_result = self.fast_upsample(fast_result.transpose(1,2).unsqueeze(2))
        fast_result = fast_result.squeeze(2).transpose(1, 2)
        fast_result = self.fast_mlp(fast_result)
        result = torch.cat([fast_result, slow_result], dim=2)
        return result
