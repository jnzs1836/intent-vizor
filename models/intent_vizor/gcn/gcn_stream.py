import torch
import torch.nn as nn
from models.intent_vizor.gcn.edge_conv import GCNeXtC, EgoGCNeXtC

class SnippetGCN(nn.Module):
    def __init__(self, feature_dim=256, k=6, gcn_groups=32, conv_groups=4, shortcut=False):
        super(SnippetGCN, self).__init__()
        self.feat_dim = feature_dim
        self.h_dim_1d = feature_dim
        self.idx_list = []
        self.gcn_groups = gcn_groups
        self.conv_groups = conv_groups
        self.shortcut = shortcut

        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=conv_groups),
            nn.BatchNorm1d(self.h_dim_1d),
            nn.ReLU(inplace=True),
            # GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        )
        self.gcn1 = GCNeXtC(self.h_dim_1d, self.h_dim_1d, k=k, groups=gcn_groups, idx=self.idx_list)

        self.gcn2 = GCNeXtC(self.h_dim_1d, self.h_dim_1d, k=k, groups=gcn_groups, idx=self.idx_list)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.backbone1[0].weight)

    def forward(self, snip_feature):
        del self.idx_list[:]  # clean the idx list  
        identity = snip_feature
        base_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        base_feature = self.gcn1(base_feature)
        gcnext_feature = self.gcn2(base_feature)  #
        out_feature = gcnext_feature + identity 
        return out_feature


class SnippetTopicGCN(nn.Module):
    def __init__(self, feature_dim=256, topic_dim=16, k=6, gcn_groups=32, conv_groups=4, shortcut=False, gcn_mode=None):
        super(SnippetTopicGCN, self).__init__()
        self.feat_dim = feature_dim
        self.h_dim_1d = feature_dim
        self.idx_list = []
        self.topic_dim = topic_dim
        self.conv_groups = conv_groups
        self.gcn_groups= gcn_groups


        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=conv_groups),
            nn.BatchNorm1d(self.h_dim_1d),
            nn.ReLU(inplace=True),
            # GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        )
        self.backbone_topic = nn.Sequential(
            nn.Conv1d(self.topic_dim, self.h_dim_1d, kernel_size=1, padding=0, groups=conv_groups),
            nn.BatchNorm1d(self.h_dim_1d),
            nn.ReLU(inplace=True)
        )
        self.gcn1 = EgoGCNeXtC(self.h_dim_1d, self.h_dim_1d, self.h_dim_1d, k=k, groups=gcn_groups, idx=self.idx_list, mode=gcn_mode)

        self.gcn2 = EgoGCNeXtC(self.h_dim_1d, self.h_dim_1d, self.h_dim_1d, k=k, groups=gcn_groups, idx=self.idx_list, mode=gcn_mode)
        self.init_weights()


    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.backbone1[0].weight)
        torch.nn.init.kaiming_normal_(self.backbone_topic[0].weight)

    def forward(self, snip_features, topic_embedding):
        del self.idx_list[:]  # clean the idx list
        identity = snip_features
        topic_embedding = topic_embedding.unsqueeze(-1)
        base_feature = self.backbone1(snip_features).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        topic_feature = self.backbone_topic(topic_embedding).contiguous()
        base_feature = self.gcn1(base_feature, topic_feature)
        gcnext_feature = self.gcn2(base_feature, topic_feature)  #
        gcnext_feature = gcnext_feature + identity
        return gcnext_feature 


class GraphStream(nn.Module):
    def __init__(self, device="cuda", topic_embedding_dim=64,
                 feature_dim=256, k=6, dropout=0.5, gcn_groups=32, conv_groups=4, gcn_shortcut=True, ego_gcn_num=1,
                 gcn_mode=None
                 ):
        nn.Module.__init__(self)
        self.device = device
        self.feature_dim = feature_dim
        self.topic_embedding_dim = topic_embedding_dim
        self.snippet_gcn = SnippetGCN(feature_dim=self.feature_dim, k=k,
                                      gcn_groups=gcn_groups, conv_groups=conv_groups, shortcut=gcn_shortcut
                                      )
        # self.snippet_topic_gcn = SnippetTopicGCN(feature_dim=self.feature_dim, topic_dim=self.topic_embedding_dim,
        #                                          k=k, gcn_groups=gcn_groups, conv_groups=conv_groups)

        self.snippet_ego_gcns= nn.ModuleList(
            [SnippetTopicGCN(feature_dim=self.feature_dim, topic_dim=self.topic_embedding_dim, shortcut=gcn_shortcut,
                                                 k=k, gcn_groups=gcn_groups, conv_groups=conv_groups, gcn_mode=gcn_mode)
             for i in range(ego_gcn_num)]
        )
        self.ego_gcn_num = ego_gcn_num
        self.merge = nn.Sequential(
            nn.Conv1d(3 * feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm1d(num_features=feature_dim),
            nn.ReLU(),
        )
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.merge[0].weight)

    def forward(self, video_features, topic_embeddings):
        batch_size = video_features.size(0)
        tmp = video_features.contiguous().view(batch_size, -1, video_features.size(-1))
        gcn_result = self.snippet_gcn(tmp.transpose(1, 2))
        gcn_result = gcn_result.transpose(1, 2)
        x = gcn_result
        for i in range(self.ego_gcn_num):
            x = self.snippet_ego_gcns[i](x.transpose(1, 2), topic_embeddings)
            x = x.transpose(1, 2)
        topic_gcn_result = x
        return topic_gcn_result
        # merged_result = torch.cat(
        #     (video_features, gcn_result, topic_gcn_result), dim=-1)

        # merged_result = self.merge(merged_result.transpose(1, 2)).transpose(1, 2)

        # return merged_result
