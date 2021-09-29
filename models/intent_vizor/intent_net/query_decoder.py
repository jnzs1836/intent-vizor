import torch
import torch.nn as nn


class QueryDecoder(nn.Module):
    def __init__(self, topic_embedding_dim=128, hidden_dim=128, query_embedding_dim=300):
        nn.Module.__init__(self)
        self.topic_embedding_dim = topic_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.topic_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * query_embedding_dim)
        )
        pass

    def forward(self, topic_embeddings):
        return self.mlp(topic_embeddings)
