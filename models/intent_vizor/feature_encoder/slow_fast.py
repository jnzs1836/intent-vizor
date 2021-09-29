import torch
import torch.nn as nn


class SlowFastFusion(nn.Module):
    # original hyper-param: 1024 deconv channel
    def __init__(self, device="cuda",
                 in_channel=2048, conv1_channel=512, conv2_channel=256,
                 slow_deconv1_channel=256, slow_deconv2_channel=128,
                 fast_deconv1_channel=128, fast_deconv2_channel=128,
                 similarity_dim=1000,
                 concept_dim=300, max_segment_num=20, max_frame_num=200,
                 topic_num = 10, topic_embedding_dim=64, hidden_dim=64, output_mlp_hidden_dim=256,
                 slow_feature_dim = 128,
                 fast_feature_dim=256, fusion_dim=128
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
        self.slow_transpose_conv1d_1 = torch.nn.ConvTranspose1d(slow_feature_dim, self.slow_deconv1_channel,
                                                                kernel_size=4,
                                                                stride=2, padding=1)
        self.slow_transpose_conv1d_2 = torch.nn.ConvTranspose1d(self.slow_deconv1_channel, self.fusion_dim,
                                                                kernel_size=5,
                                                                stride=8, padding=1, output_padding=5)

        self.fast_transpose_conv1d_1 = torch.nn.ConvTranspose1d(fast_feature_dim, self.fast_deconv1_channel, kernel_size=4,
                                                           stride=2, padding=1)
        self.fast_transpose_conv1d_2 = torch.nn.ConvTranspose1d(self.fast_deconv1_channel, self.fusion_dim, kernel_size=4,
                                                           stride=2, padding=1, output_padding=0)

        self.relu = nn.ReLU()
        self.slow_conv_post = nn.Sequential(
            nn.Conv1d(self.fusion_dim, self.fusion_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=self.fusion_dim),
            nn.ReLU(),
            nn.Conv1d(self.fusion_dim, self.fusion_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=self.fusion_dim),
            nn.ReLU(),
            nn.AvgPool1d(3, padding=1, stride=1)
        )
        self.slow_conv_post = nn.Sequential(
            nn.AvgPool1d(5, padding=2, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(5, padding=2, stride=1)
        )

        self.fast_conv_post = nn.Sequential(
            nn.Conv1d(self.fusion_dim, self.fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.fusion_dim),
            nn.ReLU(),
            nn.Conv1d(self.fusion_dim, self.fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.fusion_dim),
            nn.ReLU(),
            nn.AvgPool1d(3, padding=1, stride=1)
        )

        self.fast_conv_post = nn.Sequential(
            nn.AvgPool1d(5, padding=2, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(5, padding=2, stride=1)
        )



        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_normal_(self.slow_transpose_conv1d_1.weight)
        torch.nn.init.kaiming_normal_(self.fast_transpose_conv1d_1.weight)
        # torch.nn.init.kaiming_normal_(self.slow_conv_post[0].weight)
        # torch.nn.init.kaiming_normal_(self.slow_conv_post[3].weight)
        # torch.nn.init.kaiming_normal_(self.fast_conv_post[0].weight)
        # torch.nn.init.kaiming_normal_(self.fast_conv_post[3].weight)

    # batch tensor: batch_size * max_seg_num * max_seg_length * 2048/4096
    # seg_len list(list(int)) : batch_size * seg_num (num of frame)
    # concept : batch_size * 300
    def forward(self, frame_features, slow_result, fast_result):
        slow_result = self.slow_transpose_conv1d_1(slow_result.transpose(1, 2))
        slow_result = self.slow_transpose_conv1d_2(slow_result)
        slow_result = self.slow_conv_post(slow_result).transpose(1, 2)
        
        fast_result = self.fast_transpose_conv1d_1(fast_result.transpose(1, 2))
        fast_result = self.fast_transpose_conv1d_2(fast_result)
        fast_result = self.fast_conv_post(fast_result).transpose(1, 2)
        

        result = torch.cat([fast_result, slow_result], dim=2)
        return result

