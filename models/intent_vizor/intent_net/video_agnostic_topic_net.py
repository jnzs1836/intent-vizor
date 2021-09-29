import torch
import torch.nn as nn


class VideoAgnosticTopicNet(nn.Module):
    def __init__(self, device, topic_num,
                 concept_dim=300, hidden_dim=512, num_hidden_layer=2, feature_dim=256,
                 slow_feature_dim=256, fast_feature_dim=128, dropout=0.5,
                 feature_transformer_head=8, feature_transformer_layer=3,
                 query_attention_head=4, gcn_groups=32, gcn_conv_groups=4, k=6,
                 ego_gcn_num=1, gcn_mode=None

                 ):
        nn.Module.__init__(self)

        self.mlp = nn.Sequential(
                nn.Linear(2 * concept_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, topic_num),
                nn.Softmax(dim=1)
        )
        self.init_weight()



    def init_weight(self):
        nn.init.kaiming_normal_(self.mlp[0].weight.data)
        nn.init.kaiming_normal_(self.mlp[2].weight.data)
        nn.init.kaiming_normal_(self.mlp[4].weight.data)

    def forward(self, batch, seg_len, concept1, concept2, video_features, slow_features, fast_features):
        concepts_cat = torch.cat([concept1, concept2], dim=1)
        return self.mlp(concepts_cat)


class VideoAttentionTopicNet(nn.Module):
    def __init__(self, device, topic_num,
                 concept_dim=300, hidden_dim=512, num_hidden_layer=2, feature_dim=256,
                 slow_feature_dim=256, fast_feature_dim=128, dropout=0.5,
                 feature_transformer_head=8, feature_transformer_layer=3,
                 query_attention_head=4, gcn_groups=32, gcn_conv_groups=4, k=6,
                 ego_gcn_num=1, gcn_mode=None):
        nn.Module.__init__(self)
        self.slow_feature_dim = slow_feature_dim
        self.fast_feature_dim = fast_feature_dim
        self.slow_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                          kdim=self.slow_feature_dim, vdim=self.slow_feature_dim,
                                                          num_heads=query_attention_head, dropout=dropout)

        self.fast_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                          kdim=self.fast_feature_dim, vdim=self.fast_feature_dim,
                                                          dropout=dropout,
                                                          num_heads=query_attention_head)

        self.mlp = nn.Sequential(
            nn.Linear(2 * concept_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, topic_num),
            nn.Softmax(dim=1)
        )
        self.relu = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.mlp[0].weight.data)
        nn.init.kaiming_normal_(self.mlp[2].weight.data)
        nn.init.kaiming_normal_(self.mlp[4].weight.data)

    def forward(self, batch, seg_len, concept1, concept2, video_features, slow_features, fast_features):
        concept_cat = torch.cat([concept1, concept2])
        concept_stack = torch.stack([concept1, concept2], dim=1)
        concept_inputs = concept_stack.transpose(0, 1)
        slow_concept_result, _ = self.slow_query_attention(concept_inputs, slow_features.transpose(0, 1),
                                                           slow_features.transpose(0, 1))
        # slow_concept_result = slow_concept_result.transpose(0, 1) + concept
        slow_concept_result = slow_concept_result.transpose(0, 1)
        slow_concept_result = self.relu(slow_concept_result)
        fast_concept_result, _ = self.fast_query_attention(concept_inputs, fast_features.transpose(0, 1),
                                                           fast_features.transpose(0, 1))

        fast_concept_result = fast_concept_result.transpose(0, 1)
        fast_concept_result = self.relu(fast_concept_result)

        slow_concept_result = torch.sum(slow_concept_result, dim=1) / slow_concept_result.size(1)
        fast_concept_result = torch.sum(fast_concept_result, dim=1) / fast_concept_result.size(1)
        aggregated_embeddings = torch.cat(
            [slow_concept_result, fast_concept_result], dim=1)
        return self.mlp(aggregated_embeddings)


