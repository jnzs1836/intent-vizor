import gc
import math
import torch
import torch.nn as nn
from .score_net import ScoreNet
from .feature_encoder import FeatureEncoder
from .embedding import VariationalEmbedding, VanillaEmbedding
from models.intent_vizor.visual_query.shot_query_net import ShotQueryNet
from models.intent_vizor.intent_net.query_decoder import QueryDecoder
from models.intent_vizor.visual_query.shot_query_graph_net import ShotQueryGraphNet


class TopicAwareShotModel(nn.Module):
    def __init__(self, device="cuda",
                 topic_num=20, topic_embedding_dim=256,
                 score_net_hidden_dim=64,
                 concept_dim=300, topic_net_hidden_dim=128, topic_net_num_hidden_layer=2,
                 topic_net_feature_transformer_head=8, topic_net_feature_transformer_layer=3,
                 topic_net_query_attention_head=4,
                 max_segment_num=20, max_frame_num=200,
                 slow_feature_dim=2048, fast_feature_dim=512, fusion_dim=512,
                 dropout=0.5, k=6, local_k=6, gcn_groups=32, gcn_conv_groups=4,
                 score_net_similarity_dim=1024, score_net_similarity_module_shortcut=False,
                 score_net_mlp_hidden_dim=512,
                 score_net_mlp_num_hidden_layer=2, score_net_mlp_init_var=0.0025, score_net_norm_layer="batch",
                 threshold=0.01, fusion="dconv", topic_net="transformer", mlp_activation="relu",
                 topic_embedding_type="vanilla", topic_embedding_truncation=0,
                 score_net_gcn_num_layer=1, topic_net_gcn_num_layer=1,
                 topic_net_attention_mlp_hidden_dim=256, topic_net_attention_num_layer=3,
                 topic_net_shot_query_dim=256,
                 score_net_similarity_module="inner_product",
                 branch_type="dual", local_gcn_num_layer=1, query_decoder_hidden_dim=256,
                 topic_embedding_non_linear_mlp=False, gcn_mode=None, local_gcn_mode=None,
                 frame_feature_dim=2048,
                 local_gcn_use_pooling=False

                 ):
        nn.Module.__init__(self)
        self.device = device
        if topic_embedding_type == "vanilla":
            self.topic_embedding = VanillaEmbedding(topic_num, topic_embedding_dim)
        elif topic_embedding_type == "variational":
            self.topic_embedding = VariationalEmbedding(topic_num, topic_embedding_dim,
                                                        truncation=topic_embedding_truncation,
                                                        non_linear_mlp=topic_embedding_non_linear_mlp
                                                        )
        self.feature_encoder = FeatureEncoder(device=device,
                                              slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim)
        self.query_decoder = QueryDecoder(topic_embedding_dim=topic_embedding_dim, hidden_dim=query_decoder_hidden_dim,
                                          query_embedding_dim=concept_dim)

        self.score_net = ScoreNet(device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim,
                                  fusion_dim=fusion_dim,
                                  slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                  dropout=dropout,
                                  k=k, local_k=local_k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                                  similarity_dim=score_net_similarity_dim,
                                  similarity_module_shortcut=score_net_similarity_module_shortcut,
                                  score_mlp_num_hidden_layer=score_net_mlp_num_hidden_layer,
                                  score_mlp_hidden_dim=score_net_mlp_hidden_dim, mlp_init_var=score_net_mlp_init_var,
                                  norm_layer=score_net_norm_layer, fusion=fusion, mlp_activation=mlp_activation,
                                  ego_gcn_num=score_net_gcn_num_layer,
                                  similarity_module=score_net_similarity_module,
                                  gcn_mode=gcn_mode, local_gcn_mode=local_gcn_mode,
                                  local_gcn_use_pooling=local_gcn_use_pooling
                                  )


        if topic_net == "graph":
            self.topic_net = ShotQueryGraphNet(
                feature_dim=frame_feature_dim,
                device=device, topic_num=topic_num, concept_dim=concept_dim,
                hidden_dim=topic_net_hidden_dim,
                num_hidden_layer=topic_net_num_hidden_layer,
                slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                feature_transformer_head=topic_net_feature_transformer_head,
                feature_transformer_layer=topic_net_feature_transformer_layer,
                query_attention_head=topic_net_query_attention_head,
                ego_gcn_num=topic_net_gcn_num_layer,
                k=k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                dropout=dropout,
                gcn_mode=gcn_mode
            )
            # self.topic_net = TopicGraphNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
            #                      hidden_dim=topic_net_hidden_dim, num_hidden_layer=topic_net_num_hidden_layer,
            #                      slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
            #                      feature_transformer_head=topic_net_feature_transformer_head,
            #                      feature_transformer_layer=topic_net_feature_transformer_layer,
            #                      query_attention_head=topic_net_query_attention_head,
            #                      ego_gcn_num=topic_net_gcn_num_layer,
            #                      k=k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
            #                      dropout=dropout)
        else:
            self.topic_net = ShotQueryNet(
                topic_num=topic_num, shot_query_dim=topic_net_shot_query_dim, shot_feature_dim=2048,
                mlp_hidden_dim=topic_net_hidden_dim,
                # num_hidden_layer=topic_net_num_hidden_layer,
                slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                query_attention_head=topic_net_query_attention_head,
                dropout=dropout, attention_mlp_hidden_dim=topic_net_attention_mlp_hidden_dim,
                attention_layer_num=topic_net_attention_num_layer,

            )
        self.topic_num = topic_num
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.non_linear = nn.ReLU()
        self.shrink_ratio = 16
        self.threshold_to_set = threshold
        self.threshold = 0
        self.init_weight()

    def load_pretrained_model(self, feature_encoder, score_net, embedding_layer):
        self.feature_encoder = feature_encoder
        self.score_net = score_net
        self.topic_embedding = embedding_layer
        gc.collect()
        torch.cuda.empty_cache()

    def init_weight(self):
        pass

    def get_pad_features(self, features):
        target_len = math.ceil(features.size(1) / self.shrink_ratio) * self.shrink_ratio
        pad = torch.zeros(features.size(0), target_len - features.size(1), features.size(2)).to(
            device=features.device)
        frame_features_pad = torch.cat([features, pad], dim=1)
        return frame_features_pad
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

    def forward(self, batch, shot_query):

        batch_size = batch.size(0)
        frame_features = batch
        frame_features_pad = self.get_pad_features(features=batch)
        _, slow_features, fast_features = self.feature_encoder(batch, frame_features_pad)
        topic_probs = self.topic_net(batch, shot_query, slow_features, fast_features)
        overall_score = torch.zeros(batch_size, frame_features.size(1)).to(device=self.device)
        all_scores = []
        prior_loss = 0
        topic_batch_size = 2
        assert self.topic_num % topic_batch_size == 0
        for idx in range(self.topic_num // topic_batch_size):
            topic = torch.arange(topic_batch_size, dtype=torch.long) + idx * topic_batch_size
            topic_prob = topic_probs[:, topic]
            # topic = torch.ones(batch_size, dtype=torch.long) * topic_id
            topic = topic.to(device=self.device)
            topic_embeddings, _prior_loss = self.topic_embedding(topic)
            prior_loss += _prior_loss
            topic_score, _ = self.score_net(None, None, None, None, topic_embeddings, None, frame_features.expand(topic_batch_size, -1, -1),
                    slow_features.expand(topic_batch_size, -1, -1), fast_features.expand(topic_batch_size, -1, -1))
            # topic_score = topic_score * topic_prob
            all_scores.append(topic_score)
            topic_score = torch.matmul(topic_prob, topic_score)
            # topic_score = torch.mul(topic_score, topic_prob.view(batch_size, 1))
            topic_score -= self.threshold
            # print(topic_score.size())
            overall_score += self.non_linear(topic_score)
        overall_score /= self.topic_num
        all_scores = torch.cat(all_scores, dim=0).unsqueeze(0)
        return overall_score, {
            "topic_probs": topic_probs,
            "all_scores": all_scores,
            "prior_loss": prior_loss
        }

