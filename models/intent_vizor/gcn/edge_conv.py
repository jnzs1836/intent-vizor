# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.intent_vizor.gcn.gtad import get_graph_feature

# dynamic graph from knn


# basic block
class GCNeXtC(nn.Module):
    def __init__(self, channel_in, channel_out, k=10, norm_layer=None, groups=32, width_group=4, idx=None):
        super(GCNeXtC, self).__init__()
        self.k = k
        self.groups = groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups
        # self.tconvs = nn.Sequential(
        #     nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
        #     nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
        #     nn.Conv1d(width, channel_out, kernel_size=1),
        # ) # temporal graph
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1),
            # nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1),
            # nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        )  # temporal graph
        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1),
            # nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups),
            #nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) # semantic graph

        self.relu = nn.ReLU(True)
        self.idx_list = idx
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_normal_(self.tconvs[0].weight)
        torch.nn.init.kaiming_normal_(self.tconvs[2].weight)
        torch.nn.init.kaiming_normal_(self.tconvs[4].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[0].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[2].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[4].weight)

    def forward(self, x):
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph

        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)

        out = tout + identity + sout  # fusion

        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)

class EgoGCNeXtC(nn.Module):
    def __init__(self, channel_in, channel_out, topic_channel, k=10, norm_layer=None, groups=32, width_group=4,
                 idx=None, mode=None):
        super(EgoGCNeXtC, self).__init__()
        assert mode is not None
        self.k = k
        self.groups = groups
        self.topic_channel = topic_channel
        self.intent_mode = mode
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups

        channel_embedding = channel_in
        
        if self.intent_mode == "mutual":
            self.intent_mlp = nn.Sequential(
                nn.Conv1d(topic_channel, channel_embedding, 1),
                nn.ReLU(),
                nn.Conv1d(channel_embedding, channel_embedding, 1)
            )
            self.feature_mlp = lambda x: x
            self.channel_sconvs = channel_embedding
        elif self.intent_mode == "mutual_map":
            self.intent_mlp = nn.Sequential(
                nn.Conv1d(topic_channel, channel_embedding, 1),
                nn.ReLU(),
                # nn.Conv1d(channel_embedding, channel_embedding, 1)
            )
            self.feature_mlp = nn.Sequential(
                nn.Conv2d(2 * channel_in, channel_embedding, 1),
                nn.ReLU()
            )
            self.channel_sconvs = channel_embedding
        elif self.intent_mode == "cat":
            self.channel_sconvs = channel_in * 2 + topic_channel

        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1),
            # nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1),
            # nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        )  # temporal graph
        self.sconvs = nn.Sequential(
            nn.Conv2d(self.channel_sconvs, width, kernel_size=1),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        )  # semantic graph

        self.relu = nn.ReLU(True)
        self.idx_list = idx
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_normal_(self.tconvs[0].weight)
        torch.nn.init.kaiming_normal_(self.tconvs[2].weight)
        torch.nn.init.kaiming_normal_(self.tconvs[4].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[0].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[3].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[6].weight)
        if self.intent_mode == "mutual":
            torch.nn.init.kaiming_normal_(self.intent_mlp[0].weight)
            torch.nn.init.kaiming_normal_(self.intent_mlp[2].weight)
        if self.intent_mode == "mutual_map":
            torch.nn.init.kaiming_normal_(self.intent_mlp[0].weight)
            # torch.nn.init.kaiming_normal_(self.intent_mlp[2].weight)
            torch.nn.init.kaiming_normal_(self.feature_mlp[0].weight)

    def forward(self, x, topic_features):
        identity = x  # residual
        x_t = x
        tout = self.tconvs(x_t)  # conv on temporal graph
        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        if self.intent_mode == "cat":
            xf_pad = torch.zeros(topic_features.size(0), topic_features.size(1), x_f.size(2), x_f.size(3)).to(device=x.device)
            topic_pad = torch.zeros(topic_features.size(0), x_f.size(1), 1, 1).to(device=x.device)
            topic_features = topic_features.unsqueeze(-1)
            topic_features_padded = torch.cat([topic_pad, topic_features], dim=1)
            topic_features_padded = topic_features_padded.expand(-1, -1, x_f.size(2), 1)
            x_f_padded = torch.cat([x_f, xf_pad], dim=1)
            merged_graph = torch.cat([x_f_padded, topic_features_padded], dim=3)
        elif self.intent_mode == "mutual" or self.intent_mode == "mutual_map":
            x_f_mapped = self.feature_mlp(x_f)
            topic_mapped = self.intent_mlp(topic_features)
            topic_mapped = topic_mapped.unsqueeze(-1)
            topic_mapped = topic_mapped.expand(-1, -1, x_f.size(2), 1)
            merged_graph = torch.cat([x_f_mapped, topic_mapped], dim=3)
        else:
            raise Exception("Invalid GCN mode")
        sout = self.sconvs(merged_graph)  # conv on semantic graph
        # sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)
        sout = sout.mean(dim=-1, keepdim=False)
        out = tout + identity + sout  # fusion
        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)


class EgoPartiteGNeXtC(nn.Module):
    def __init__(self, channel_in, channel_out, topic_channel, k=10, norm_layer=None, groups=32, width_group=4,
                 idx=None, mode=None):
        super(EgoPartiteGNeXtC, self).__init__()
        assert mode is not None
        self.k = k
        self.groups = groups
        self.topic_channel = topic_channel
        self.intent_mode = mode
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups

        channel_embedding = channel_in

        if self.intent_mode == "mutual":
            self.intent_mlp = nn.Sequential(
                nn.Conv1d(topic_channel, channel_embedding, 1),
                nn.ReLU(),
                nn.Conv1d(channel_embedding, channel_embedding, 1)
            )
            self.feature_mlp = lambda x: x
            self.channel_sconvs = channel_embedding
        elif self.intent_mode == "mutual_map":
            self.intent_mlp = nn.Sequential(
                nn.Conv1d(topic_channel, channel_embedding, 1),
                nn.ReLU(),
                # nn.Conv1d(channel_embedding, channel_embedding, 1)
            )
            self.feature_mlp = nn.Sequential(
                nn.Conv2d(2 * channel_in, channel_embedding, 1),
                nn.ReLU()
            )
            self.channel_sconvs = channel_embedding
        elif self.intent_mode == "cat":
            self.channel_sconvs = channel_in * 2 + topic_channel

        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1),
            # nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1),
            # nn.BatchNorm1d(width),
            nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        )  # temporal graph
        self.sconvs = nn.Sequential(
            nn.Conv2d(self.channel_sconvs, width, kernel_size=1),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        )  # semantic graph

        self.relu = nn.ReLU(True)
        self.idx_list = idx
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_normal_(self.tconvs[0].weight)
        torch.nn.init.kaiming_normal_(self.tconvs[2].weight)
        torch.nn.init.kaiming_normal_(self.tconvs[4].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[0].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[3].weight)
        torch.nn.init.kaiming_normal_(self.sconvs[6].weight)
        if self.intent_mode == "mutual":
            torch.nn.init.kaiming_normal_(self.intent_mlp[0].weight)
            torch.nn.init.kaiming_normal_(self.intent_mlp[2].weight)
        if self.intent_mode == "mutual_map":
            torch.nn.init.kaiming_normal_(self.intent_mlp[0].weight)
            # torch.nn.init.kaiming_normal_(self.intent_mlp[2].weight)
            torch.nn.init.kaiming_normal_(self.feature_mlp[0].weight)

    def forward(self, x, topic_features):
        identity = x  # residual
        x_t = x
        tout = self.tconvs(x_t)  # conv on temporal graph
        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        if self.intent_mode == "cat":
            xf_pad = torch.zeros(topic_features.size(0), topic_features.size(1), x_f.size(2), x_f.size(3)).to(
                device=x.device)
            topic_pad = torch.zeros(topic_features.size(0), x_f.size(1), 1, 1).to(device=x.device)
            topic_features = topic_features.unsqueeze(-1)
            topic_features_padded = torch.cat([topic_pad, topic_features], dim=1)
            topic_features_padded = topic_features_padded.expand(-1, -1, x_f.size(2), 1)
            x_f_padded = torch.cat([x_f, xf_pad], dim=1)
            merged_graph = torch.cat([x_f_padded, topic_features_padded], dim=3)
        elif self.intent_mode == "mutual" or self.intent_mode == "mutual_map":
            x_f_mapped = self.feature_mlp(x_f)
            topic_mapped = self.intent_mlp(topic_features)
            topic_mapped = topic_mapped.unsqueeze(-2)
            topic_mapped = topic_mapped.expand(-1, -1, x_f.size(2), -1)
            merged_graph = torch.cat([x_f_mapped, topic_mapped], dim=3)
        else:
            raise Exception("Invalid GCN mode")
        sout = self.sconvs(merged_graph)  # conv on semantic graph
        # sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)
        sout = sout.mean(dim=-1, keepdim=False)
        out = tout + identity + sout  # fusion
        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)
