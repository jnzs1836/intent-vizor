import torch
import torch.nn as nn


class ShotAttention(nn.Module):
    def __init__(self, query_dim=256, key_dim=256, attention_head=8, mlp_hidden_dim=256, dropout=0):
        nn.Module.__init__(self)
        self.attention_layer = nn.MultiheadAttention(embed_dim=query_dim,
                                                          kdim=key_dim, vdim=key_dim,
                                                          num_heads=attention_head, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(query_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, query_dim)
        )

    def forward(self, query, key):
        identity = query
        x = query.transpose(0, 1)
        x, _ = self.attention_layer(x, key.transpose(0, 1), key.transpose(0, 1))
        x = x.transpose(0, 1)
        x = self.mlp(x)
        return x + identity


class ShotQueryNet(nn.Module):
    def __init__(self, slow_feature_dim=256, fast_feature_dim=256, shot_feature_dim=2048,shot_query_dim=256,
                query_attention_head=8, dropout=0, attention_mlp_hidden_dim=256, attention_layer_num=1, mlp_hidden_dim=256,
                 topic_num=20
                 ):
        nn.Module.__init__(self)
        self.query_linear = nn.Linear(shot_feature_dim, shot_query_dim)
        self.slow_attention_layers = nn.ModuleList(
            [ShotAttention(shot_query_dim, slow_feature_dim, query_attention_head,
                                                   mlp_hidden_dim=attention_mlp_hidden_dim, dropout=dropout) for i in range(attention_layer_num)]
        )
        self.fast_attention_layers = nn.ModuleList(
            [ShotAttention(shot_query_dim, fast_feature_dim, query_attention_head,
                                                   mlp_hidden_dim=attention_mlp_hidden_dim, dropout=dropout) for i in range(attention_layer_num)]
        )
        self.attention_layer_num = attention_layer_num

        self.mlp = nn.Sequential(
            nn.Linear(3 * shot_query_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, topic_num),
            nn.Softmax(dim=1)
        )

    def forward(self, shots, video_features, slow_features, fast_features):
        shot_query = self.query_linear(shots)
        slow = shot_query
        fast = shot_query
        for i in range(self.attention_layer_num):
            slow = self.slow_attention_layers[i](slow, slow_features)
            fast = self.fast_attention_layers[i](fast, fast_features)
        fast, _ = torch.max(fast, dim=1)
        slow, _ = torch.max(slow, dim=1)
        shot_query = torch.sum(shot_query, dim=1) / shot_query.size(1)
        merged = torch.cat([fast, slow, shot_query], dim=1)
        return self.mlp(merged)
