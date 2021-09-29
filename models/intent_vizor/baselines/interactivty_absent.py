import math
import torch
import torch.nn as nn
from models.intent_vizor.score_net import ScoreNet
from models.intent_vizor.intent_net.concept_net import ConceptNet
from models.intent_vizor.feature_encoder import FeatureEncoder
class TopicAbsentModel(nn.Module):
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
                 score_net_mlp_num_hidden_layer=2,

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
                                  score_mlp_hidden_dim=score_net_mlp_hidden_dim
                                  )
        self.concept_net = ConceptNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
                                  hidden_dim=topic_net_hidden_dim, num_hidden_layer=topic_net_num_hidden_layer,
                                  slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                  feature_transformer_head=topic_net_feature_transformer_head,
                                  feature_transformer_layer=topic_net_feature_transformer_layer,
                                  query_attention_head=topic_net_query_attention_head, topic_embedding_dim=topic_embedding_dim,
                                  dropout=dropout)
        self.topic_num = topic_num
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.non_linear = nn.ReLU()
        self.shrink_ratio = 16
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

    def forward(self, batch, seg_len, concept1, concept2):
        batch_size = batch.size(0)
        frame_features, frame_features_pad = self.get_frame_features(batch, seg_len)
        video_features, slow_features, fast_features = self.feature_encoder(batch, frame_features_pad, seg_len, concept1, concept2)
        topic_embeddings = self.concept_net(batch, seg_len, concept1, concept2, video_features, slow_features, fast_features)
        topic_score, _ = self.score_net(batch, seg_len, concept1, concept2, topic_embeddings, video_features, frame_features, slow_features, fast_features)
        overall_score = topic_score
        return overall_score, overall_score
