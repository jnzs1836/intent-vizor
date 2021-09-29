import torch
import torch.nn as nn
import math
from models.intent_vizor.gcn.gcn_stream import SnippetTopicGCN


class BranchLocalGCN(nn.Module):
    def __init__(self, frame_feature_dim, slow_feature_dim, fast_feature_dim, fusion_dim, k,
                 gcn_groups, conv_groups, shrink_ratio=16, use_slow_branch=False, use_fast_branch=True,
                 local_gcn_num_layer=1, local_gcn_mode=None
                 ):
        nn.Module.__init__(self)

        self.use_slow_branch = use_slow_branch
        self.use_fast_branch = use_fast_branch
        self.gcn_num = local_gcn_num_layer
        if use_slow_branch:
            self.slow_ego_gcns = nn.ModuleList(
                [
                    SnippetTopicGCN(
                        feature_dim=fusion_dim, topic_dim=fusion_dim,
                        k=k if k < 16 else 16, gcn_groups=gcn_groups, conv_groups=conv_groups,
                        gcn_mode=local_gcn_mode
                    ) for i in range(local_gcn_num_layer)
                ]
            )
            self.slow_mlp = nn.Sequential(
                nn.Linear(slow_feature_dim, fusion_dim)
            )

        # self.slow_ego_gcn = SnippetTopicGCN(
        #     feature_dim=fusion_dim, topic_dim=fusion_dim,
        #     k=k if k < 16 else 16, gcn_groups=gcn_groups, conv_groups=conv_groups
        # )
        if use_fast_branch:
            self.fast_ego_gcns = nn.ModuleList(
                [
                    SnippetTopicGCN(
                        feature_dim=fusion_dim, topic_dim=fusion_dim,
                        k=k if k < 4 else 4, gcn_groups=gcn_groups, conv_groups=conv_groups,
                        gcn_mode=local_gcn_mode
                    ) for i in range(local_gcn_num_layer)
                ]
            )
            self.fast_mlp = nn.Sequential(
                nn.Linear(fast_feature_dim, fusion_dim)
            )
        # self.fast_ego_gcn = SnippetTopicGCN(
        #     feature_dim=fusion_dim, topic_dim=fusion_dim,
        #     k=k if k < 4 else 4, gcn_groups=gcn_groups, conv_groups=conv_groups
        # )

        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_feature_dim, fusion_dim)
        )
        self.shrink_ratio = shrink_ratio

    def forward(self, frame_features, slow_result, fast_result):
        batch_size = frame_features.size(0)

        frame_preprocessed = self.frame_mlp(frame_features)
        target_len = math.ceil(frame_features.size(1) / self.shrink_ratio) * self.shrink_ratio
        pad = torch.zeros(batch_size, target_len - frame_features.size(1), frame_preprocessed.size(2)).to(
            device=frame_features.device)
        frame_preprocessed_pad = torch.cat([frame_preprocessed, pad], dim=1).contiguous()

        result_list = []

        if self.use_fast_branch:
            fast_preprocessed = self.fast_mlp(fast_result).contiguous().view(batch_size * fast_result.size(1), -1)
            fast_frame_features = frame_preprocessed_pad.view(fast_preprocessed.size(0), -1,
                                                              frame_preprocessed.size(-1))
            x = fast_frame_features.transpose(1, 2)
            for i in range(self.gcn_num):
                x = self.fast_ego_gcns[i](x, fast_preprocessed)
            fast_processed = x
            fast_processed = fast_processed.transpose(1, 2).contiguous().view(batch_size, -1, fast_processed.size(1))
            result_list.append(fast_processed)

        if self.use_slow_branch:
            slow_preprocessed = self.slow_mlp(slow_result).contiguous().view(batch_size * slow_result.size(1), -1)
            slow_frame_features = frame_preprocessed_pad.view(slow_preprocessed.size(0), -1, frame_preprocessed.size(-1))
            y = slow_frame_features.transpose(1, 2)
            for i in range(self.gcn_num):
                y = self.slow_ego_gcns[i](y, slow_preprocessed)
            slow_processed = y
            slow_processed = slow_processed.transpose(1, 2).contiguous().view(batch_size, -1, slow_processed.size(1))
            result_list.append(slow_processed)

        result = torch.cat(result_list, dim=2)
        return result
