import torch
import torch.nn as nn
import math
from models.intent_vizor.gcn.gcn_stream import SnippetTopicGCN
from models.intent_vizor.utils import TransposeModule


class LocalGCN(nn.Module):
    def __init__(self, frame_feature_dim, slow_feature_dim, fast_feature_dim, fusion_dim, k,
                 gcn_groups, conv_groups, shrink_ratio=16, local_gcn_mode=None, use_pooling=False
                 ):
        nn.Module.__init__(self)
        self.slow_ego_gcn = SnippetTopicGCN(
            feature_dim=fusion_dim, topic_dim=fusion_dim,
            k=k if k < 16 else 16, gcn_groups=gcn_groups, conv_groups=conv_groups, gcn_mode=local_gcn_mode
        )
        self.fast_ego_gcn = SnippetTopicGCN(
            feature_dim=fusion_dim, topic_dim=fusion_dim,
            k=k if k < 4 else 4, gcn_groups=gcn_groups, conv_groups=conv_groups, gcn_mode=local_gcn_mode
        )

        fast_mlp_layers = [
            nn.Linear(fast_feature_dim, fusion_dim),
        ]
        slow_mlp_layers = [
            nn.Linear(slow_feature_dim, fusion_dim)
        ]
        if use_pooling:
            fast_mlp_layers.extend(
                [
                    TransposeModule(),
                    nn.AvgPool1d(kernel_size=9, padding=4, stride=1),
                    TransposeModule()
                ]
            )
            slow_mlp_layers.extend([
                TransposeModule(),
                nn.AvgPool1d(kernel_size=3, padding=1, stride=1),
                TransposeModule()
            ])
        self.slow_mlp = nn.Sequential(
            *slow_mlp_layers
        )
        self.fast_mlp = nn.Sequential(
            *fast_mlp_layers
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_feature_dim, fusion_dim)
        )
        self.shrink_ratio = shrink_ratio

    def forward(self, frame_features, slow_result, fast_result):
        batch_size = frame_features.size(0)
        slow_preprocessed = self.slow_mlp(slow_result).contiguous().view(batch_size * slow_result.size(1), -1)
        fast_preprocessed = self.fast_mlp(fast_result).contiguous().view(batch_size * fast_result.size(1), -1)
        frame_preprocessed = self.frame_mlp(frame_features)
        target_len = math.ceil(frame_features.size(1) / self.shrink_ratio) * self.shrink_ratio
        pad = torch.zeros(batch_size, target_len - frame_features.size(1), frame_preprocessed.size(2)).to(
            device=frame_features.device)
        frame_preprocessed_pad = torch.cat([frame_preprocessed, pad], dim=1).contiguous()

        fast_frame_features = frame_preprocessed_pad.view(fast_preprocessed.size(0), -1, frame_preprocessed.size(-1))
        slow_frame_features = frame_preprocessed_pad.view(slow_preprocessed.size(0), -1, frame_preprocessed.size(-1))
        slow_processed = self.slow_ego_gcn(slow_frame_features.transpose(1, 2), slow_preprocessed)
        slow_processed = slow_processed.transpose(1, 2).contiguous().view(batch_size, -1, slow_processed.size(1))
        fast_processed = self.fast_ego_gcn(fast_frame_features.transpose(1, 2), fast_preprocessed)
        fast_processed = fast_processed.transpose(1, 2).contiguous().view(batch_size, -1, fast_processed.size(1))
        result = torch.cat([fast_processed, slow_processed], dim=2)
        return result
