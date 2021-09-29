import torch
import torch.nn as nn
from models.intent_vizor.score_net.attention import Attention
from models.intent_vizor.gcn.gcn_stream import SnippetGCN
from models.intent_vizor.gcn.edge_conv import EgoPartiteGNeXtC
class SnippetShotQueryGCN(nn.Module):
    def __init__(self, feature_dim=256, topic_dim=16, k=6, gcn_groups=32, conv_groups=4, shortcut=False, gcn_mode=None):
        super(SnippetShotQueryGCN, self).__init__()
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
        self.gcn1 = EgoPartiteGNeXtC(self.h_dim_1d, self.h_dim_1d, self.h_dim_1d, k=k, groups=gcn_groups, idx=self.idx_list, mode=gcn_mode)

        self.gcn2 = EgoPartiteGNeXtC(self.h_dim_1d, self.h_dim_1d, self.h_dim_1d, k=k, groups=gcn_groups, idx=self.idx_list, mode=gcn_mode)
        self.init_weights()


    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.backbone1[0].weight)
        torch.nn.init.kaiming_normal_(self.backbone_topic[0].weight)

    def forward(self, snip_features, topic_embedding):
        del self.idx_list[:]  # clean the idx list
        identity = snip_features
        topic_embedding = topic_embedding.transpose(1, 2)
        base_feature = self.backbone1(snip_features).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        topic_feature = self.backbone_topic(topic_embedding).contiguous()
        base_feature = self.gcn1(base_feature, topic_feature)
        gcnext_feature = self.gcn2(base_feature, topic_feature)  #
        gcnext_feature = gcnext_feature + identity
        return gcnext_feature


class ShotQueryGraphStream(nn.Module):
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
            [SnippetShotQueryGCN(feature_dim=self.feature_dim, topic_dim=self.topic_embedding_dim, shortcut=gcn_shortcut,
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


class ShotQueryGraphNet(nn.Module):
    def __init__(self, device, topic_num,
                 concept_dim=300, hidden_dim=512, num_hidden_layer=2, feature_dim=256,
                 slow_feature_dim=256, fast_feature_dim=128, dropout=0.5,
                 feature_transformer_head=8, feature_transformer_layer=3,
                 query_attention_head=4, gcn_groups=32, gcn_conv_groups=4, k=6,
                 ego_gcn_num=1, gcn_mode=None

                 ):
        nn.Module.__init__(self)

        self.mlp = self.make_topic_mlp(feature_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim,
                                       topic_num, dropout)
        self.slow_feature_dim = slow_feature_dim
        self.fast_feature_dim = fast_feature_dim
        self.feature_dim = feature_dim
        self.device=device
        self.self_attention = Attention(self.feature_dim, self.feature_dim, self.feature_dim)

        self.fast_stream = ShotQueryGraphStream(feature_dim=fast_feature_dim, topic_embedding_dim=feature_dim, k=k,
                                       dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )
        self.slow_stream = ShotQueryGraphStream(feature_dim=slow_feature_dim, topic_embedding_dim=feature_dim,
                                       k=k, dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )


        self.slow_query_attention = nn.MultiheadAttention(embed_dim=feature_dim,
                                                          kdim=self.slow_feature_dim, vdim=self.slow_feature_dim,
                                                          num_heads=query_attention_head, dropout=dropout)

        self.fast_query_attention = nn.MultiheadAttention(embed_dim=feature_dim,
                                                          kdim=self.fast_feature_dim, vdim=self.fast_feature_dim,
                                                          dropout=dropout,
                                                          num_heads=query_attention_head)

        self.relu = nn.ReLU()
        self.init_weight()

    def make_topic_mlp(self, concept_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim, topic_num, dropout):
        layers = []
        layers.append(nn.Linear(concept_dim * 2 + slow_feature_dim + fast_feature_dim, hidden_dim))
        nn.init.normal_(layers[-1].weight.data, 0, 1)
        layers.append(nn.BatchNorm1d(num_features=hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for i in range(num_hidden_layer - 1):
            if i == 0:
                linear_input = hidden_dim
            else:
                linear_input = hidden_dim // 2
            layers.append(nn.Linear(linear_input, hidden_dim // 2))
            nn.init.normal_(layers[-1].weight.data, 0, 1)
            layers.append(nn.BatchNorm1d(num_features=hidden_dim // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        # add the last layer
        output_linear = nn.Linear(hidden_dim // 2, topic_num)
        # nn.init.normal_(output_linear.weight.data, 0, 1)
        layers.append(output_linear)
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def init_weight(self):
        nn.init.normal_(self.mlp[0].weight.data, 0, 1)
        nn.init.normal_(self.mlp[4].weight.data, 0, 1)

    def forward(self, batch, shot_query, slow_features, fast_features):
        slow_attention_inputs = slow_features
        fast_attention_inputs = fast_features

        slow_attention_result = self.slow_stream(slow_attention_inputs, shot_query)
        slow_attention_agg = torch.sum(slow_attention_result, dim=1) / slow_attention_result.size(1)

        fast_attention_result = self.fast_stream(fast_attention_inputs, shot_query)
        fast_attention_agg = torch.sum(fast_attention_result, dim=1) / fast_features.size(1)

        shot_query_inputs = shot_query.transpose(0, 1)
        slow_concept_result, _ = self.slow_query_attention(shot_query_inputs, slow_attention_result.transpose(0, 1),
                                                           slow_attention_result.transpose(0, 1))
        # slow_concept_result = slow_concept_result.transpose(0, 1) + concept
        slow_concept_result = slow_concept_result.transpose(0, 1)
        slow_concept_result = self.relu(slow_concept_result)
        fast_concept_result, _ = self.fast_query_attention(shot_query_inputs, fast_attention_result.transpose(0, 1),
                                                           fast_attention_result.transpose(0, 1))

        fast_concept_result = fast_concept_result.transpose(0, 1)
        fast_concept_result = self.relu(fast_concept_result)
        # fast_concept_result = fast_concept_result.transpose(0, 1) + concept

        slow_concept_result = torch.sum(slow_concept_result, dim=1) / slow_concept_result.size(1)
        fast_concept_result = torch.sum(fast_concept_result, dim=1) / fast_concept_result.size(1)
        
        aggregated_embeddings = torch.cat([slow_attention_agg, fast_attention_agg, slow_concept_result, fast_concept_result], dim=1)
        return self.mlp(aggregated_embeddings)
