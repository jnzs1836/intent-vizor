import torch.nn as nn
from models.intent_vizor.gcn.gcn_stream import GraphStream
from models.intent_vizor.feature_encoder.slow_fast import SlowFastFusion
from models.intent_vizor.score_net.attention_fusion import AttentionFusion
from models.intent_vizor.score_net.similarity_module import SimilarityModule, SimilarityAbsentModule
from models.intent_vizor.score_net.upsampling import UpSampling
from models.intent_vizor.score_net.branch_local_gcn import BranchLocalGCN
from models.intent_vizor.score_net import TransposeModule


def make_score_mlp(num_hidden_layer, hidden_dim, input_dim, mlp_activation="relu", dropout=0, norm_layer="batch"):
    layers = []
    activation_class = nn.ReLU if mlp_activation == "relu" else nn.ReLU6
    for i in range(num_hidden_layer):
        if i == 0:
            layer_input_dim = input_dim
        else:
            layer_input_dim = hidden_dim
        layers.append(nn.Linear(layer_input_dim, hidden_dim))
        # nn.init.normal_(layers[-1].weight.data, 0, 1)
        nn.init.kaiming_normal_(layers[-1].weight.data)
        if norm_layer == "batch":
            layers.append(TransposeModule())
            layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            layers.append(TransposeModule())
        else:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_class())
        layers.append(nn.Dropout(p=dropout))

    # add the last layer
    output_linear_input_dim = hidden_dim if num_hidden_layer > 0 else input_dim
    output_linear = nn.Linear(output_linear_input_dim, 1)
    nn.init.kaiming_normal_(output_linear.weight.data)
    # nn.init.normal_(output_linear.weight.data, 0, self.mlp_init_var)
    layers.append(output_linear)
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class BranchScoreNet(nn.Module):
    def __init__(self, device="cuda",
                 in_channel=2048, similarity_dim=1024,
                 concept_dim=300, max_segment_num=20, max_frame_num=200,
                 topic_num=10, topic_embedding_dim=64,
                 slow_feature_dim=256, fast_feature_dim=128, fusion_dim=128, dropout=0.5, k=6, local_k=6,
                 similarity_module_shortcut=False, score_mlp_num_hidden_layer=2, score_mlp_hidden_dim=512,
                 gcn_groups=32, gcn_conv_groups=4, mlp_init_var=0.0025, norm_layer="batch", fusion="dconv", mlp_activation="relu",
                 ego_gcn_num=1, similarity_module="inner_product", use_slow_branch=False, use_fast_branch=False,
                 local_gcn_num_layer=1, local_gcn_mode=None, gcn_mode=None, output_mode="score"
                 ):
        nn.Module.__init__(self)
        self.device = device
        self.in_channel = in_channel
        self.concept_dim = concept_dim
        self.similarity_dim = similarity_dim
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.fusion_dim = fusion_dim
        self.topic_embedding_dim = topic_embedding_dim
        self.dropout = dropout
        self.mlp_init_var = mlp_init_var
        self.output_mode = output_mode
        # self.score_mlp = nn.Sequential(
        #     nn.Linear(self.similarity_dim, self.similarity_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(self.similarity_dim // 2, self.similarity_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(self.similarity_dim // 2, 1),
        #     nn.Sigmoid()
        # )

        assert use_slow_branch or use_fast_branch

        self.use_slow_branch = use_slow_branch
        self.use_fast_branch = use_fast_branch

        self.fusion_type = fusion
        self.mlp_activation = mlp_activation

        self.norm_layer = norm_layer
        self.score_mlp = self.make_score_mlp(score_mlp_num_hidden_layer, score_mlp_hidden_dim, similarity_dim)

        if self.use_fast_branch:
            self.fast_stream = GraphStream(feature_dim=fast_feature_dim, topic_embedding_dim=topic_embedding_dim, k=k,
                                       dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )
        if self.use_slow_branch:
            self.slow_stream = GraphStream(feature_dim=slow_feature_dim, topic_embedding_dim=topic_embedding_dim,
                                       k=k, dropout=dropout, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                                       ego_gcn_num=ego_gcn_num, gcn_mode=gcn_mode
                                       )

        branch_num = 0
        if use_slow_branch:
            branch_num += 1
        if use_fast_branch:
            branch_num += 1

        if self.fusion_type == "attention":
            self.slow_fast_fusion = AttentionFusion(slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                               fusion_dim=fusion_dim)
        elif self.fusion_type == "upsample":
            self.slow_fast_fusion = UpSampling(slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                               fusion_dim=fusion_dim)
        elif self.fusion_type == "local_gcn":
            self.slow_fast_fusion = BranchLocalGCN(
                frame_feature_dim=in_channel,
                slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                fusion_dim=fusion_dim, gcn_groups=gcn_groups, conv_groups=gcn_conv_groups,
                k=local_k, use_fast_branch=use_fast_branch, use_slow_branch=use_slow_branch,
                local_gcn_num_layer=local_gcn_num_layer, local_gcn_mode=local_gcn_mode
            )
        else:
            self.slow_fast_fusion = SlowFastFusion(slow_feature_dim=slow_feature_dim, fast_feature_dim=fast_feature_dim,
                                               fusion_dim=fusion_dim)

        if similarity_module == "inner_product":
            self.similarity_module = SimilarityModule(branch_num*self.fusion_dim, topic_embedding_dim, similarity_dim,
                                                  shortcut=similarity_module_shortcut)
        elif similarity_module == "absent":
            self.similarity_module = SimilarityAbsentModule(branch_num*self.fusion_dim, topic_embedding_dim, similarity_dim,
                                                  shortcut=similarity_module_shortcut)

        self.init_weight()

    def make_score_mlp(self, num_hidden_layer, hidden_dim, input_dim):
        layers = []
        activation_class = nn.ReLU if self.mlp_activation == "relu" else nn.ReLU6
        for i in range(num_hidden_layer):
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_dim
            layers.append(nn.Linear(layer_input_dim, hidden_dim))
            # nn.init.normal_(layers[-1].weight.data, 0, 1)
            nn.init.kaiming_normal_(layers[-1].weight.data)
            if self.norm_layer == "batch":
                layers.append(TransposeModule())
                layers.append(nn.BatchNorm1d(num_features=hidden_dim))
                layers.append(TransposeModule())
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_class())
            layers.append(nn.Dropout(p=self.dropout))

        # add the last layer
        output_linear_input_dim = hidden_dim if num_hidden_layer > 0 else input_dim
        output_linear = nn.Linear(output_linear_input_dim, 1)
        nn.init.kaiming_normal_(output_linear.weight.data)
        # nn.init.normal_(output_linear.weight.data, 0, self.mlp_init_var)
        layers.append(output_linear)
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def init_weight(self):
        pass
        # nn.init.normal_(self.score_mlp[0].weight.data, 0, 1)
        # nn.init.normal_(self.score_mlp[3].weight.data, 0, 1)


    def forward(self,batch, seg_len,concept1,concept2, topic_embeddings, video_features, frame_features, slow_features, fast_features):
        #
        fast_result = None
        slow_result = None
        if self.use_fast_branch:
            fast_result = self.fast_stream(fast_features, topic_embeddings)

        if self.use_slow_branch:
            slow_result = self.slow_stream(slow_features, topic_embeddings)


        result = self.slow_fast_fusion(frame_features, slow_result, fast_result)
        result = result[:, :frame_features.size(1),:]

        topic_similar = self.similarity_module(result, topic_embeddings)
        if self.output_mode == "similarity":
            return topic_similar, topic_similar
        # topic_similar = topic_similar / 50
        overall_score = self.score_mlp(topic_similar)
        overall_score = overall_score.squeeze(dim=-1)
        return overall_score, overall_score


class LateFusionScoreNet(nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        kwargs['use_slow_branch'] = True
        kwargs['use_fast_branch'] = False
        self.slow_branch = BranchScoreNet(*args, **kwargs)

        kwargs['use_slow_branch'] = False
        kwargs['use_fast_branch'] = True
        self.fast_branch = BranchScoreNet(*args, **kwargs)

    def forward(self, batch, seg_len,concept1,concept2, topic_embeddings, video_features, frame_features, slow_features, fast_features):
        slow_scores, _ = self.slow_branch(batch, seg_len,concept1,concept2, topic_embeddings, video_features, frame_features, slow_features, fast_features)
        fast_scores, _ = self.fast_branch(batch, seg_len,concept1,concept2, topic_embeddings, video_features, frame_features, slow_features, fast_features)
        overall_scores = slow_scores + fast_scores
        return overall_scores, overall_scores


class MiddleFusionScoreNet(nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        kwargs['use_slow_branch'] = True
        kwargs['use_fast_branch'] = False
        self.slow_branch = BranchScoreNet(*args, **kwargs, output_mode="similarity")

        kwargs['use_slow_branch'] = False
        kwargs['use_fast_branch'] = True
        self.fast_branch = BranchScoreNet(*args, **kwargs, output_mode="similarity")
        self.score_mlp = make_score_mlp(kwargs['score_mlp_num_hidden_layer'], kwargs['score_mlp_hidden_dim'],
                                        kwargs['similarity_dim'])

    def forward(self, batch, seg_len, concept1, concept2, topic_embeddings, video_features, frame_features,
                slow_features, fast_features):
        slow_topic_similar, _ = self.slow_branch(batch, seg_len, concept1, concept2, topic_embeddings, video_features,
                                          frame_features, slow_features, fast_features)
        fast_topic_similar, _ = self.fast_branch(batch, seg_len, concept1, concept2, topic_embeddings, video_features,
                                          frame_features, slow_features, fast_features)
        topic_similar = slow_topic_similar + fast_topic_similar
        overall_scores = self.score_mlp(topic_similar)
        overall_scores = overall_scores.squeeze(dim=-1)
        return overall_scores, overall_scores
