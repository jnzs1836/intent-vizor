import torch
import torch.nn as nn


class SimilarityAbsentModule(nn.Module):
    def __init__(self, result_dim, topic_embedding_dim, similarity_dim, shortcut=False):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(result_dim, result_dim),
            nn.ReLU(),
            nn.Linear(result_dim, similarity_dim),
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.mlp[0].weight.data)
        nn.init.kaiming_normal_(self.mlp[2].weight.data)

    def forward(self, result, topic_embeddings):
        return self.mlp(result)


class SimilarityModule(nn.Module):
    def __init__(self, result_dim, topic_embedding_dim, similarity_dim, shortcut=False):
        nn.Module.__init__(self)
        self.result_dim = result_dim
        self.topic_embedding_dim = topic_embedding_dim
        self.similarity_dim = similarity_dim
        self.shortcut = shortcut

        self.similarity_linear1 = torch.nn.Linear(self.result_dim, self.similarity_dim, bias=False)
        # self.batchnorm1 = torch.nn.BatchNorm1d(num_features=self.similarity_dim)
        self.similarity_linear2 = torch.nn.Linear(self.topic_embedding_dim, self.similarity_dim, bias=False)
        # self.batchnorm2 = torch.nn.BatchNorm1d(num_features=self.similarity_dim)
        self.init_weight()

    # Function: forward
    # result: (batch_size, seq_len, result_dim)
    # topic_embeddings: (batch_size, topic_embedding_dim)
    # output: (batch_size, similarity_dim)
    def forward(self, result, topic_embeddings):
        similar_1 = self.similarity_linear1(result)
        # similar_1 = self.batchnorm1(similar_1.transpose(1, 2)).transpose(1, 2)
        topic_similar = self.similarity_linear2(topic_embeddings)
        # topic_similar = self.batchnorm2(topic_similar)
        topic_similar = topic_similar.unsqueeze(1) * similar_1
        if self.shortcut:
            topic_similar += similar_1
        return topic_similar

    def init_weight(self):
        nn.init.kaiming_normal_(self.similarity_linear1.weight.data)
        nn.init.kaiming_normal_(self.similarity_linear2.weight.data)
