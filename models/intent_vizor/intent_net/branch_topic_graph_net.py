import torch
import torch.nn as nn
from models.intent_vizor.score_net.attention import Attention
from models.intent_vizor.gcn.gcn_stream import GraphStream


class BranchTopicGraphNet(nn.Module):
    def __init__(self, device, topic_num,
                 concept_dim=300, hidden_dim=512, num_hidden_layer=2, feature_dim=256,
                 slow_feature_dim=256, fast_feature_dim=128, dropout=0.5,
                 feature_transformer_head=8, feature_transformer_layer=3,
                 query_attention_head=4, gcn_groups=32, gcn_conv_groups=4, k=6,
                 ego_gcn_num=1, use_slow_branch=False, use_fast_branch=False,
                 gcn_mode=None

                 ):
        nn.Module.__init__(self)
        assert use_slow_branch or use_fast_branch

        self.use_slow_branch = use_slow_branch
        self.use_fast_branch = use_fast_branch

        self.mlp = self.make_topic_mlp(concept_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim,
                                       topic_num, dropout)
        self.slow_feature_dim = slow_feature_dim
        self.fast_feature_dim = fast_feature_dim
        self.feature_dim = feature_dim
        self.device=device
        self.self_attention = Attention(self.feature_dim, self.feature_dim, self.feature_dim)



        if self.use_fast_branch:
            self.fast_stream = GraphStream(feature_dim=fast_feature_dim, topic_embedding_dim=concept_dim * 2, k=k,
                                       dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )
            self.fast_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                              kdim=self.fast_feature_dim, vdim=self.fast_feature_dim,
                                                              dropout=dropout,
                                                              num_heads=query_attention_head)

        if self.use_slow_branch:
            self.slow_stream = GraphStream(feature_dim=slow_feature_dim, topic_embedding_dim=concept_dim * 2,
                                       k=k, dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )

            self.slow_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                          kdim=self.slow_feature_dim, vdim=self.slow_feature_dim,
                                                          num_heads=query_attention_head, dropout=dropout)

        self.relu = nn.ReLU()
        self.init_weight()

    def make_topic_mlp(self, concept_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim, topic_num, dropout):
        layers = []
        input_dim = 0
        if self.use_fast_branch:
            input_dim += concept_dim + fast_feature_dim
        if self.use_slow_branch:
            input_dim += concept_dim + slow_feature_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
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

    def forward(self, batch, seg_len, concept1, concept2, video_features, slow_features, fast_features):
        slow_attention_inputs = slow_features
        fast_attention_inputs = fast_features

        concept_cat = torch.cat([concept1, concept2], dim=1)
        concept_stack = torch.stack([concept1, concept2], dim=1)
        concept_inputs = concept_stack.transpose(0, 1)

        result_list = []

        if self.use_slow_branch:
            slow_attention_result = self.slow_stream(slow_attention_inputs, concept_cat)
            slow_attention_agg = torch.sum(slow_attention_result, dim=1) / slow_attention_result.size(1)
            slow_concept_result, _ = self.slow_query_attention(concept_inputs, slow_attention_result.transpose(0, 1),
                                                           slow_attention_result.transpose(0, 1))
            slow_concept_result = slow_concept_result.transpose(0, 1)
            slow_concept_result = self.relu(slow_concept_result)
            slow_concept_result = torch.sum(slow_concept_result, dim=1) / slow_concept_result.size(1)
            result_list.append(slow_attention_agg)
            result_list.append(slow_concept_result)

        if self.use_fast_branch:
            fast_attention_result = self.fast_stream(fast_attention_inputs, concept_cat)
            fast_attention_agg = torch.sum(fast_attention_result, dim=1) / fast_features.size(1)
            fast_concept_result, _ = self.fast_query_attention(concept_inputs, fast_attention_result.transpose(0, 1),
                                                               fast_attention_result.transpose(0, 1))
            fast_concept_result = fast_concept_result.transpose(0, 1)
            fast_concept_result = self.relu(fast_concept_result)
            fast_concept_result = torch.sum(fast_concept_result, dim=1) / fast_concept_result.size(1)
            result_list.append(fast_attention_agg)
            result_list.append(fast_concept_result)
            # fast_concept_result = fast_concept_result.transpose(0, 1) + concept

        aggregated_embeddings = torch.cat(result_list, dim=1)

        return self.mlp(aggregated_embeddings)
