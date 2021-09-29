import math
import torch
import torch.nn as nn
from .score_net import ScoreNet
from models.intent_vizor.intent_net.topic_net import TopicNet
from models.intent_vizor.intent_net.topic_graph_net import TopicGraphNet
from .feature_encoder import FeatureEncoder
from scipy.stats import truncnorm


class VanillaEmbedding(nn.Module):
    def __init__(self, topic_num, topic_embedding_dim):
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(topic_num, topic_embedding_dim)
        self.init_weights()


    def init_weights(self):
        nn.init.normal_(self.embedding.weight.data, 0, 1)
    def forward(self, topic_ids):
        embeddings = self.embedding(topic_ids)
        return embeddings, 0


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    values = torch.Tensor(values)
    return values



class VariationalEmbedding(nn.Module):
    def __init__(self, topic_num, topic_embedding_dim, truncation=0, non_linear_mlp=False, use_mlp=True):
        nn.Module.__init__(self)
        self.soft_plus = nn.Softplus()
        self.mu_embedding = nn.Embedding(topic_num, topic_embedding_dim)
        self.var_embedding = nn.Embedding(topic_num, topic_embedding_dim)

        self.mlp = lambda x: x
        self.non_linear_mlp = non_linear_mlp
        if non_linear_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(topic_embedding_dim, topic_embedding_dim, bias=False),
                nn.ReLU(),
                nn.Linear(topic_embedding_dim, topic_embedding_dim, bias=False),
            )


        self.truncation = truncation

        self.init_weights()

    def reparameterize(self, mu, log_variance):
        std = torch.exp(0.5 * log_variance)
        rand =torch.randn(std.size())
        if not self.training and self.truncation > 0:
            rand = truncated_normal(std.size(), self.truncation)
        epsilon = torch.autograd.Variable(rand).to(log_variance.device)
        return (mu + epsilon * std)

    def init_weights(self):
        nn.init.kaiming_normal_(self.mu_embedding.weight.data)
        nn.init.kaiming_normal_(self.var_embedding.weight.data)
        if self.non_linear_mlp:
            nn.init.kaiming_normal_(self.mlp[0].weight.data)
        # nn.init.kaiming_normal_(self.mlp[2].weight.data)
            nn.init.normal_(self.mlp[-1].weight.data, 0, 1)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def forward(self, topic_ids):
        mu = self.mu_embedding(topic_ids)
        log_variance = torch.log(self.soft_plus(self.var_embedding(topic_ids)))
        h = self.reparameterize(mu, log_variance)
        h = self.mlp(h)
        prior_loss = self.prior_loss(mu, log_variance)
        return h, prior_loss

class VariationalTopicAwareModel(nn.Module):
    def __init__(self, device="cuda",
                 topic_num=20, topic_embedding_dim=256,
                 score_net_hidden_dim=64,
                 concept_dim=300, topic_net_hidden_dim=128, topic_net_num_hidden_layer=2,
                 topic_net_feature_transformer_head=8, topic_net_feature_transformer_layer=3,
                 topic_net_query_attention_head=4,
                 max_segment_num=20, max_frame_num=200,
                 slow_feature_dim=2048, fast_feature_dim=512, fusion_dim=512,
                 dropout=0.5, k=6, gcn_groups=32, gcn_conv_groups=4,
                 score_net_similarity_dim=1024, score_net_similarity_module_shortcut=False,
                 score_net_mlp_hidden_dim=512,
                 score_net_mlp_num_hidden_layer=2, score_net_mlp_init_var=0.0025, score_net_norm_layer="batch",
                 threshold=0.01, fusion="dconv", topic_net="transformer", mlp_activation="relu"

                 ):
        nn.Module.__init__(self)
        self.device = device
        self.topic_embedding = nn.Embedding(topic_num, topic_embedding_dim)
        self.feature_encoder = FeatureEncoder(device=device,
                                              slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim)
        self.score_net = ScoreNet(device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim,
                                  fusion_dim=fusion_dim,
                                  slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                  dropout=dropout,
                                  k=k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                                  similarity_dim=score_net_similarity_dim,
                                  similarity_module_shortcut=score_net_similarity_module_shortcut,
                                  score_mlp_num_hidden_layer=score_net_mlp_num_hidden_layer,
                                  score_mlp_hidden_dim=score_net_mlp_hidden_dim, mlp_init_var=score_net_mlp_init_var,
                                  norm_layer=score_net_norm_layer, fusion=fusion, mlp_activation=mlp_activation
                                  )

        if topic_net == "graph":
            self.topic_net = TopicGraphNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
                                  hidden_dim=topic_net_hidden_dim, num_hidden_layer=topic_net_num_hidden_layer,
                                  slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                  feature_transformer_head=topic_net_feature_transformer_head,
                                  feature_transformer_layer=topic_net_feature_transformer_layer,
                                  query_attention_head=topic_net_query_attention_head,
                                  k=k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                                  dropout=dropout)
        else:
            self.topic_net = TopicNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
                                  hidden_dim=topic_net_hidden_dim, num_hidden_layer=topic_net_num_hidden_layer,
                                  slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                  feature_transformer_head=topic_net_feature_transformer_head,
                                  feature_transformer_layer=topic_net_feature_transformer_layer,
                                  query_attention_head=topic_net_query_attention_head,
                                  dropout=dropout)
        self.topic_num = topic_num
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.non_linear = nn.ReLU()
        self.shrink_ratio = 16
        self.threshold_to_set = threshold
        self.threshold = 0
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.topic_embedding.weight.data, 0, 1)

    def get_frame_features(self, batch, seg_len):
        mask=torch.zeros(batch.size(0), batch.size(1),batch.size(2),dtype=torch.bool).to(device=self.device)
        for i in range(seg_len.size(0)):
            for j in range(len(seg_len[i])):
                for k in range(seg_len[i][j]):
                    mask[i][j][k]=1
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, batch.size(3))
        frame_features = batch.masked_select(mask)
        frame_features = frame_features.view( batch.size(0), -1, batch.size(3))
        target_len = math.ceil(frame_features.size(1) / self.shrink_ratio ) * self.shrink_ratio
        pad = torch.zeros(batch.size(0), target_len - frame_features.size(1), frame_features.size(2)).to(device=batch.device)
        frame_features_pad = torch.cat([frame_features, pad], dim=1)
        return frame_features, frame_features_pad

    def activate_non_linearity(self):
        self.threshold = self.threshold_to_set

    def forward(self, batch, seg_len, concept1, concept2):
        batch_size = batch.size(0)
        frame_features, frame_features_pad = self.get_frame_features(batch, seg_len)
        video_features, slow_features, fast_features = self.feature_encoder(batch, frame_features_pad, seg_len, concept1, concept2)
        topic_probs = self.topic_net(batch, seg_len, concept1, concept2, video_features, slow_features, fast_features)

        overall_score = torch.zeros(batch_size, frame_features.size(1)).to(device=self.device)
        all_scores = []
        for topic_id in range(self.topic_num):
            topic_prob = topic_probs[:, topic_id]
            topic = torch.ones(batch_size, dtype=torch.long) * topic_id
            topic = topic.to(device=self.device)
            topic_embeddings = self.topic_embedding(topic)
            topic_score, _ = self.score_net(batch, seg_len, concept1, concept2, topic_embeddings, video_features, frame_features, slow_features, fast_features)
            # topic_score = topic_score * topic_prob
            all_scores.append(topic_score)
            topic_score = torch.mul(topic_score, topic_prob.view(batch_size, 1))
            topic_score -= self.threshold
            # print(topic_score.size())
            overall_score += self.non_linear(topic_score)
        overall_score /= self.topic_num
        all_scores = torch.stack(all_scores, dim=1)
        return overall_score, {
            "topic_probs": topic_probs,
            "all_scores": all_scores
        }

