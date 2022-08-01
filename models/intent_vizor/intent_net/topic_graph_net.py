import math
import torch
import torch.nn as nn
from ..score_net.attention import Attention
from ..score_net import ScoreNet
from ..gcn.gcn_stream import GraphStream


class IntentDropout(nn.Module):
    def __init__(self, p=0):
        nn.Module.__init__(self)
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        _, indices = torch.sort(x, descending=True)
        indices = indices.detach()
        indices = indices[:, :self.p]
        # max_index = torch.argmax(x, dim=1).detach()
        mask = torch.ones_like(x)
        # mask = torch.ones_like(x).scatter_(0, indices, 0).detach()
        for i in range(indices.size(0)):
            for j in range(indices.size(1)):
                x[i, indices[i, j]] = -1000
        # print(indices.cpu())
        # print(x)
        x = mask * x
        # print(indices)
        # print(x)
        return x




class TopicGraphNet(nn.Module):
    def __init__(self, device, topic_num,
                 concept_dim=300, hidden_dim=512, num_hidden_layer=2, feature_dim=256,
                 slow_feature_dim=256, fast_feature_dim=128, dropout=0.5,
                 feature_transformer_head=8, feature_transformer_layer=3,
                 query_attention_head=4, gcn_groups=32, gcn_conv_groups=4, k=6,
                 ego_gcn_num=1, gcn_mode=None, intent_dropout=0

                 ):
        nn.Module.__init__(self)
        # self.mlp = nn.Sequential(
        #     nn.Linear(concept_dim * 2 + slow_feature_dim + fast_feature_dim, hidden_dim),
        #     nn.BatchNorm1d(num_features=hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.BatchNorm1d(num_features=hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(hidden_dim // 2, topic_num),
        #     nn.Softmax(dim=1)
        # )

        self.intent_dropout_rate = intent_dropout
        self.mlp = self.make_topic_mlp(concept_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim,
                                       topic_num, dropout)
        self.slow_feature_dim = slow_feature_dim
        self.fast_feature_dim = fast_feature_dim
        self.feature_dim = feature_dim
        self.device=device
        self.self_attention = Attention(self.feature_dim, self.feature_dim, self.feature_dim)

        self.fast_stream = GraphStream(feature_dim=fast_feature_dim, topic_embedding_dim=concept_dim * 2, k=k,
                                       dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )
        self.slow_stream = GraphStream(feature_dim=slow_feature_dim, topic_embedding_dim=concept_dim * 2,
                                       k=k, dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )


        self.slow_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                          kdim=self.slow_feature_dim, vdim=self.slow_feature_dim,
                                                          num_heads=query_attention_head, dropout=dropout)

        self.fast_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                          kdim=self.fast_feature_dim, vdim=self.fast_feature_dim,
                                                          dropout=dropout,
                                                          num_heads=query_attention_head)

        self.relu = nn.ReLU()
        self.init_weight()

    def make_topic_mlp(self, concept_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim, topic_num, dropout):
        layers = []
        layers.append(nn.Linear(concept_dim * 2 + slow_feature_dim + fast_feature_dim, hidden_dim, bias=False))
        nn.init.normal_(layers[-1].weight.data, 0, 1)
        layers.append(nn.BatchNorm1d(num_features=hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for i in range(num_hidden_layer - 1):
            if i == 0:
                linear_input = hidden_dim
            else:
                linear_input = hidden_dim // 2
            layers.append(nn.Linear(linear_input, hidden_dim // 2, bias=False))
            nn.init.normal_(layers[-1].weight.data, 0, 1)
            layers.append(nn.BatchNorm1d(num_features=hidden_dim // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        # add the last layer
        output_linear = nn.Linear(hidden_dim // 2, topic_num, bias=False)
        # nn.init.normal_(output_linear.weight.data, 0, 1)
        layers.append(output_linear)
        layers.append(IntentDropout(p=self.intent_dropout_rate))
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def init_weight(self):
        nn.init.normal_(self.mlp[0].weight.data, 0, 1)
        nn.init.normal_(self.mlp[4].weight.data, 0, 1)

    def forward(self, batch, seg_len, concept1, concept2, video_features, slow_features, fast_features):
        slow_attention_inputs = slow_features
        fast_attention_inputs = fast_features

        concept_cat = torch.cat([concept1, concept2], dim=1)
        concept_stack = torch.stack([concept1, concept2], dim=1)
        slow_attention_result = self.slow_stream(slow_attention_inputs, concept_cat)
        slow_attention_agg = torch.sum(slow_attention_result, dim=1) / slow_attention_result.size(1)

        fast_attention_result = self.fast_stream(fast_attention_inputs, concept_cat)
        fast_attention_agg = torch.sum(fast_attention_result, dim=1) / fast_features.size(1)

        concept_inputs = concept_stack.transpose(0, 1)
        slow_concept_result, _ = self.slow_query_attention(concept_inputs, slow_attention_result.transpose(0, 1),
                                                           slow_attention_result.transpose(0, 1))
        # slow_concept_result = slow_concept_result.transpose(0, 1) + concept
        slow_concept_result = slow_concept_result.transpose(0, 1)
        slow_concept_result = self.relu(slow_concept_result)
        fast_concept_result, _ = self.fast_query_attention(concept_inputs, fast_attention_result.transpose(0, 1),
                                                           fast_attention_result.transpose(0, 1))

        fast_concept_result = fast_concept_result.transpose(0, 1)
        fast_concept_result = self.relu(fast_concept_result)
        # fast_concept_result = fast_concept_result.transpose(0, 1) + concept

        slow_concept_result = torch.sum(slow_concept_result, dim=1) / slow_concept_result.size(1)
        fast_concept_result = torch.sum(fast_concept_result, dim=1) / fast_concept_result.size(1)
        
        aggregated_embeddings = torch.cat([slow_attention_agg, fast_attention_agg, slow_concept_result, fast_concept_result], dim=1)
        result = self.mlp(aggregated_embeddings)
        # print(result)
        return result

