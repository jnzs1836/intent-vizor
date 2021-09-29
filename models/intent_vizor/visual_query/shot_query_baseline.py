import torch
import torch.nn as nn

class ShotLinearBaseline(nn.Module):
    def __init__(self, device="cuda",
                 frame_feature_dim=2048, num_heads=8, mlp_hidden_dim=1024,
                 ):
        nn.Module.__init__(self)

        self.mlp = nn.Sequential(
            nn.Linear(frame_feature_dim + 5 * frame_feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, batch, shot_query):
        shot_query = shot_query.unsqueeze(1)
        shot_query = shot_query.view(shot_query.size(0), 1, -1)
        shot_query_expand = shot_query.expand(-1, batch.size(1), -1)
        features = torch.cat(batch, shot_query_expand, dim=2)
        scores = self.mlp(features)
        return scores, {
            "topic_probs": None,
            "all_scores": None,
            "prior_loss": 0
        }

class ShotQueryBaseline(nn.Module):
    def __init__(self, device="cuda",
                 frame_feature_dim=2048, num_heads=8, mlp_hidden_dim=1024,
                 ):
        nn.Module.__init__(self)
        self.query_attention = nn.MultiheadAttention(embed_dim=frame_feature_dim, num_heads=num_heads,
                                                     kdim=frame_feature_dim, vdim=frame_feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(frame_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, batch, shot_query):
        batch_inputs = batch.transpose(0, 1)
        query_inputs = shot_query.transpose(0, 1)
        result, _ = self.query_attention(batch_inputs, query_inputs, query_inputs)
        result = result.transpose(0, 1)
        scores = self.mlp(result)
        scores = scores.squeeze(-1)
        return scores, {
                    "topic_probs": None,
                    "all_scores": None,
                    "prior_loss": 0
                }
