import torch
import torch.nn as nn
from models.intent_vizor.score_net.attention import Attention


class ConceptNet(nn.Module):
    def __init__(self, device, topic_num,
                 concept_dim=300, hidden_dim=512, num_hidden_layer=2, feature_dim=256,
                 slow_feature_dim=256, fast_feature_dim=128, dropout=0.5,
                 feature_transformer_head=8, feature_transformer_layer=3,
                 query_attention_head=4, topic_embedding_dim=256

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
        self.mlp = self.make_topic_mlp(concept_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim,
                                       topic_embedding_dim, dropout)
        self.slow_feature_dim = slow_feature_dim
        self.fast_feature_dim = fast_feature_dim
        self.feature_dim = feature_dim
        self.device=device
        self.self_attention = Attention(self.feature_dim, self.feature_dim, self.feature_dim)
        slow_transformer_layer = nn.TransformerEncoderLayer(d_model=self.slow_feature_dim,
                                                            nhead=feature_transformer_head,
                                                            dropout=dropout)
        self.slow_transformer = nn.TransformerEncoder(slow_transformer_layer, num_layers=feature_transformer_layer)
        fast_transformer_layer = nn.TransformerEncoderLayer(d_model=self.fast_feature_dim,
                                                            nhead=feature_transformer_head, dropout=dropout)
        self.fast_transformer = nn.TransformerEncoder(fast_transformer_layer, num_layers=feature_transformer_layer)

        self.slow_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                          kdim=self.slow_feature_dim, vdim=self.slow_feature_dim,
                                                          num_heads=query_attention_head, dropout=dropout)

        self.fast_query_attention = nn.MultiheadAttention(embed_dim=concept_dim,
                                                          kdim=self.fast_feature_dim, vdim=self.fast_feature_dim,
                                                          dropout=dropout,
                                                          num_heads=query_attention_head)

        self.init_weight()

    def make_topic_mlp(self, concept_dim, slow_feature_dim, fast_feature_dim, num_hidden_layer, hidden_dim, topic_embedding_dim, dropout):
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
        output_linear = nn.Linear(hidden_dim // 2, topic_embedding_dim)
        # nn.init.normal_(output_linear.weight.data, 0, 1)
        layers.append(output_linear)
        return nn.Sequential(*layers)

    def init_weight(self):
        nn.init.normal_(self.mlp[0].weight.data, 0, 1)
        nn.init.normal_(self.mlp[4].weight.data, 0, 1)

    def forward(self, batch, seg_len, concept1, concept2, video_features, slow_features, fast_features):
        slow_attention_inputs = slow_features.transpose(0, 1)
        fast_attention_inputs = fast_features.transpose(0, 1)

        slow_attention_result = self.slow_transformer(slow_attention_inputs).transpose(0, 1)
        slow_attention_agg = torch.sum(slow_attention_result, dim=1) / slow_attention_result.size(1)

        fast_attention_result = self.fast_transformer(fast_attention_inputs).transpose(0, 1)
        fast_attention_agg = torch.sum(fast_attention_result, dim=1) / fast_features.size(1)

        concept = torch.stack([concept1, concept2], dim=1)
        concept_inputs = concept.transpose(0, 1)
        slow_concept_result, _ = self.slow_query_attention(concept_inputs, slow_attention_result.transpose(0, 1),
                                                           slow_attention_result.transpose(0, 1))
        slow_concept_result = slow_concept_result.transpose(0, 1) + concept
        fast_concept_result, _ = self.fast_query_attention(concept_inputs, fast_attention_result.transpose(0, 1),
                                                           fast_attention_result.transpose(0, 1))
        fast_concept_result = fast_concept_result.transpose(0, 1) + concept
        slow_concept_result = torch.sum(slow_concept_result, dim=1) / slow_concept_result.size(1)
        fast_concept_result = torch.sum(fast_concept_result, dim=1) / fast_concept_result.size(1)

        aggregated_embeddings = torch.cat([slow_attention_agg, fast_attention_agg, slow_concept_result, fast_concept_result], dim=1)
        return self.mlp(aggregated_embeddings)
