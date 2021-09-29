import math
import torch
import torch.nn as nn
from .attention import Attention


class AttentionFusion(nn.Module):
    # original hyper-param: 1024 deconv channel
    def __init__(self, device="cuda",
                 in_channel=2048, conv1_channel=512, conv2_channel=256,
                 slow_deconv1_channel=256, slow_deconv2_channel=128,
                 fast_deconv1_channel=128, fast_deconv2_channel=128,
                 similarity_dim=1000,
                 concept_dim=300, max_segment_num=20, max_frame_num=200,
                 topic_num = 10, topic_embedding_dim=64, hidden_dim=64, output_mlp_hidden_dim=256,
                 slow_feature_dim = 128,
                 fast_feature_dim=256, fusion_dim=128, frame_feature_dim=2048, num_heads=8
                 ):
        nn.Module.__init__(self)
        self.device = device
        self.in_channel = in_channel
        self.conv1_channel = conv1_channel
        self.conv2_channel = conv2_channel
        self.slow_deconv1_channel = fusion_dim * 4
        self.slow_deconv2_channel = slow_deconv2_channel
        self.fast_deconv1_channel = fusion_dim * 2
        self.fast_deconv2_channel = fast_deconv2_channel
        self.concept_dim = concept_dim
        self.similarity_dim = similarity_dim
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.convs_channel = conv1_channel
        self.slow_conv_channel = slow_feature_dim
        self.fast_conv_channel = fast_feature_dim
        self.fusion_dim = fusion_dim
        self.slow_attention = nn.MultiheadAttention(frame_feature_dim, num_heads=num_heads, kdim=slow_feature_dim, vdim=slow_feature_dim)
        self.fast_attention = nn.MultiheadAttention(frame_feature_dim, num_heads=num_heads, kdim=fast_feature_dim, vdim=fast_feature_dim)
        self.mlp = nn.Sequential(
                    nn.Linear(frame_feature_dim * 2, fusion_dim * 2),
                    nn.ReLU(),
                    nn.Linear(fusion_dim * 2,2 * fusion_dim),
                    nn.ReLU()
                )
        self.relu = nn.ReLU()

        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_normal_(self.mlp[0].weight.data)
        torch.nn.init.kaiming_normal_(self.mlp[2].weight.data)
        # torch.nn.init.kaiming_normal_(self.slow_attention.weight)
        # torch.nn.init.kaiming_normal_(self.fast_attention.weight)

    # batch tensor: batch_size * max_seg_num * max_seg_length * 2048/4096
    # seg_len list(list(int)) : batch_size * seg_num (num of frame)
    # concept : batch_size * 300
    def forward(self, frame_features, slow_result, fast_result):
        query = frame_features.transpose(0, 1)
        fast_key_value = fast_result.transpose(0, 1)
        slow_key_value = slow_result.transpose(0, 1)
        slow_result, _ = self.slow_attention(query, slow_key_value, slow_key_value)
        slow_result = self.relu(slow_result.transpose(0, 1))
        fast_result, _ = self.fast_attention(query, fast_key_value, fast_key_value)
        fast_result = self.relu(fast_result.transpose(0, 1))
        result = torch.cat([fast_result, slow_result], dim=2)
        result = self.mlp(result)
        return result

