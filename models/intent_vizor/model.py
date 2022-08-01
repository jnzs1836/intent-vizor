import math
import torch
import torch.nn as nn
from .score_net.attention import Attention
from .score_net.score_net import ScoreNet
from .score_net.branch_score_net import BranchScoreNet, LateFusionScoreNet, MiddleFusionScoreNet
from .intent_net.topic_net import TopicNet
from .intent_net.topic_graph_net import TopicGraphNet
from .intent_net.branch_topic_graph_net import BranchTopicGraphNet
from .feature_encoder import FeatureEncoder
from .embedding import VariationalEmbedding, VanillaEmbedding
from .intent_net.query_decoder import QueryDecoder
from .intent_net.video_agnostic_topic_net import VideoAttentionTopicNet, VideoAgnosticTopicNet
from .feature_encoder.plain_feature import PlainFeatureEncoder


class TopicAwareModel(nn.Module):
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
                 score_net_gcn_num_layer=1, topic_net_gcn_num_layer=1, score_net_similarity_module="inner_product",
                 branch_type="dual", local_gcn_num_layer=1, query_decoder_hidden_dim=256,
                 topic_embedding_non_linear_mlp=False, gcn_mode=None, local_gcn_mode=None, local_gcn_use_pooling=False,
                 intent_dropout=0, score_branch_net="gcn", topic_branch_net="transformer", feature_encoder="fast_slow"

                 ):
        nn.Module.__init__(self)
        self.device = device

        self.use_fast_branch = False
        self.use_slow_branch = False
        if branch_type == "dual":
            self.use_fast_branch = True
            self.use_slow_branch = True
        elif branch_type == "fast_only":
            self.use_fast_branch = True
        elif branch_type == "slow_only":
            self.use_slow_branch = True
        elif branch_type == "fast_slow":
            self.use_fast_branch = True
            self.use_slow_branch = True
        elif branch_type == "late_fusion" or branch_type == "middle_fusion":
            self.use_fast_branch = True
            self.use_slow_branch = True
        else:
            raise Exception("Invalid branch setup.")

        if topic_embedding_type == "vanilla":
            self.topic_embedding = VanillaEmbedding(topic_num, topic_embedding_dim)
        elif topic_embedding_type == "variational":
            self.topic_embedding = VariationalEmbedding(topic_num, topic_embedding_dim,
                                                        truncation=topic_embedding_truncation,
                                                        non_linear_mlp=topic_embedding_non_linear_mlp)

        if feature_encoder == "fast_slow":
            self.feature_encoder = FeatureEncoder(device=device,
                                              slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim)
        elif feature_encoder == "plain":
            self.feature_encoder = PlainFeatureEncoder()

        self.query_decoder = QueryDecoder(topic_embedding_dim=topic_embedding_dim, hidden_dim=query_decoder_hidden_dim,
                                          query_embedding_dim=concept_dim)
        if branch_type == "dual":
            self.score_net = ScoreNet(device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim,
                                      fusion_dim=fusion_dim,
                                      slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                      dropout=dropout,
                                      k=k, local_k=local_k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                                      similarity_dim=score_net_similarity_dim,
                                      similarity_module_shortcut=score_net_similarity_module_shortcut,
                                      score_mlp_num_hidden_layer=score_net_mlp_num_hidden_layer,
                                      score_mlp_hidden_dim=score_net_mlp_hidden_dim,
                                      mlp_init_var=score_net_mlp_init_var,
                                      norm_layer=score_net_norm_layer, fusion=fusion, mlp_activation=mlp_activation,
                                      ego_gcn_num=score_net_gcn_num_layer,
                                      similarity_module=score_net_similarity_module,
                                      gcn_mode=gcn_mode, local_gcn_mode=local_gcn_mode,
                                      local_gcn_use_pooling=local_gcn_use_pooling
                                      )
        elif branch_type == "late_fusion":
            self.score_net = LateFusionScoreNet(
                device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim,
                fusion_dim=fusion_dim,
                slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                dropout=dropout,
                k=k, local_k=local_k, gcn_groups=gcn_groups,
                gcn_conv_groups=gcn_conv_groups,
                similarity_dim=score_net_similarity_dim,
                similarity_module_shortcut=score_net_similarity_module_shortcut,
                score_mlp_num_hidden_layer=score_net_mlp_num_hidden_layer,
                score_mlp_hidden_dim=score_net_mlp_hidden_dim,
                mlp_init_var=score_net_mlp_init_var,
                norm_layer=score_net_norm_layer, fusion=fusion,
                mlp_activation=mlp_activation,
                ego_gcn_num=score_net_gcn_num_layer,
                similarity_module=score_net_similarity_module,
                use_fast_branch=self.use_fast_branch, use_slow_branch=self.use_slow_branch,
                local_gcn_num_layer=local_gcn_num_layer,
                gcn_mode=gcn_mode, local_gcn_mode=local_gcn_mode
            ),
        elif branch_type == "middle_fusion":
            self.score_net = MiddleFusionScoreNet(
                device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim,
                fusion_dim=fusion_dim,
                slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                dropout=dropout,
                k=k, local_k=local_k, gcn_groups=gcn_groups,
                gcn_conv_groups=gcn_conv_groups,
                similarity_dim=score_net_similarity_dim,
                similarity_module_shortcut=score_net_similarity_module_shortcut,
                score_mlp_num_hidden_layer=score_net_mlp_num_hidden_layer,
                score_mlp_hidden_dim=score_net_mlp_hidden_dim,
                mlp_init_var=score_net_mlp_init_var,
                norm_layer=score_net_norm_layer, fusion=fusion,
                mlp_activation=mlp_activation,
                ego_gcn_num=score_net_gcn_num_layer,
                similarity_module=score_net_similarity_module,
                use_fast_branch=self.use_fast_branch, use_slow_branch=self.use_slow_branch,
                local_gcn_num_layer=local_gcn_num_layer,
                gcn_mode=gcn_mode, local_gcn_mode=local_gcn_mode
            )
        else:
            self.score_net = BranchScoreNet(device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim,
                                            fusion_dim=fusion_dim,
                                            slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                            dropout=dropout,
                                            k=k, local_k=local_k, gcn_groups=gcn_groups,
                                            gcn_conv_groups=gcn_conv_groups,
                                            similarity_dim=score_net_similarity_dim,
                                            similarity_module_shortcut=score_net_similarity_module_shortcut,
                                            score_mlp_num_hidden_layer=score_net_mlp_num_hidden_layer,
                                            score_mlp_hidden_dim=score_net_mlp_hidden_dim,
                                            mlp_init_var=score_net_mlp_init_var,
                                            norm_layer=score_net_norm_layer, fusion=fusion,
                                            mlp_activation=mlp_activation,
                                            ego_gcn_num=score_net_gcn_num_layer,
                                            similarity_module=score_net_similarity_module,
                                            use_fast_branch=self.use_fast_branch, use_slow_branch=self.use_slow_branch,
                                            local_gcn_num_layer=local_gcn_num_layer, local_gcn_mode=local_gcn_mode,
                                            gcn_mode=gcn_mode, branch_net=score_branch_net
                                            )

        if topic_net == "graph":
            if branch_type == "dual":
                self.topic_net = TopicGraphNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
                                               hidden_dim=topic_net_hidden_dim,
                                               num_hidden_layer=topic_net_num_hidden_layer,
                                               slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                               feature_transformer_head=topic_net_feature_transformer_head,
                                               feature_transformer_layer=topic_net_feature_transformer_layer,
                                               query_attention_head=topic_net_query_attention_head,
                                               ego_gcn_num=topic_net_gcn_num_layer,
                                               k=k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                                               dropout=dropout,
                                               gcn_mode=gcn_mode,
                                               intent_dropout=intent_dropout
                                               )
            else:
                self.topic_net = BranchTopicGraphNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
                                                     hidden_dim=topic_net_hidden_dim,
                                                     num_hidden_layer=topic_net_num_hidden_layer,
                                                     slow_feature_dim=slow_feature_dim,
                                                     fast_feature_dim=fast_feature_dim,
                                                     feature_transformer_head=topic_net_feature_transformer_head,
                                                     feature_transformer_layer=topic_net_feature_transformer_layer,
                                                     query_attention_head=topic_net_query_attention_head,
                                                     ego_gcn_num=topic_net_gcn_num_layer,
                                                     k=k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                                                     dropout=dropout,
                                                     use_slow_branch=self.use_slow_branch,
                                                     use_fast_branch=self.use_fast_branch,
                                                     gcn_mode=gcn_mode, branch_net=topic_branch_net
                                                     )
        elif topic_net == "video_agnostic":
            self.topic_net = VideoAgnosticTopicNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
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
        elif topic_net == "video_attention":
            self.topic_net = VideoAttentionTopicNet(device=device, topic_num=topic_num, concept_dim=concept_dim,
                                                    hidden_dim=topic_net_hidden_dim,
                                                    num_hidden_layer=topic_net_num_hidden_layer,
                                                    slow_feature_dim=slow_feature_dim,
                                                    fast_feature_dim=fast_feature_dim,
                                                    feature_transformer_head=topic_net_feature_transformer_head,
                                                    feature_transformer_layer=topic_net_feature_transformer_layer,
                                                    query_attention_head=topic_net_query_attention_head,
                                                    ego_gcn_num=topic_net_gcn_num_layer,
                                                    k=k, gcn_groups=gcn_groups, gcn_conv_groups=gcn_conv_groups,
                                                    dropout=dropout,
                                                    gcn_mode=gcn_mode
                                                    )
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
        self.topic_batch_size = 2
        self.init_weight()

    def init_weight(self):
        pass

    def get_frame_features(self, batch, seg_len):
        mask = torch.zeros(batch.size(0), batch.size(1), batch.size(2), dtype=torch.bool).to(device=self.device)
        for i in range(seg_len.size(0)):
            for j in range(len(seg_len[i])):
                for k in range(seg_len[i][j]):
                    mask[i][j][k] = 1
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, batch.size(3))
        frame_features = batch.masked_select(mask)
        frame_features = frame_features.view(batch.size(0), -1, batch.size(3))
        target_len = math.ceil(frame_features.size(1) / self.shrink_ratio) * self.shrink_ratio
        pad = torch.zeros(batch.size(0), target_len - frame_features.size(1), frame_features.size(2)).to(
            device=batch.device)
        frame_features_pad = torch.cat([frame_features, pad], dim=1)
        return frame_features, frame_features_pad

    def activate_non_linearity(self):
        self.threshold = self.threshold_to_set

    def forward_decoder(self, batch, seg_len, concept1, concept2):
        batch_size = batch.size(0)
        frame_features, frame_features_pad = self.get_frame_features(batch, seg_len)
        _, slow_features, fast_features = self.feature_encoder(batch, frame_features_pad, seg_len, concept1, concept2)
        topic_probs = self.topic_net(batch, seg_len, concept1, concept2, None, slow_features, fast_features)
        topic = torch.arange(self.topic_num, dtype=torch.long)
        topic = topic.to(device=self.device)
        topic_embeddings, _prior_loss = self.topic_embedding(topic)
        topic_probs = nn.ReLU()(topic_probs - 0.05)
        topic_encodings = torch.matmul(topic_probs, topic_embeddings)
        decoded_query = self.query_decoder(topic_encodings)
        return decoded_query

    def forward(self, batch, seg_len, concept1, concept2):
        batch_size = batch.size(0)
        frame_features, frame_features_pad = self.get_frame_features(batch, seg_len)
        _, slow_features, fast_features = self.feature_encoder(batch, frame_features_pad, seg_len, concept1, concept2)
        topic_probs = self.topic_net(batch, seg_len, concept1, concept2, None, slow_features, fast_features)

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
            topic_score, _ = self.score_net(batch, seg_len, concept1, concept2, topic_embeddings, None,
                                            frame_features.expand(topic_batch_size, -1, -1),
                                            slow_features.expand(topic_batch_size, -1, -1),
                                            fast_features.expand(topic_batch_size, -1, -1))
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
