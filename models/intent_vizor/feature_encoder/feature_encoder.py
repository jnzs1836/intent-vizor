import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(self, device="cuda",
                 in_channel=2048, conv1_channel=512, conv2_channel=256,
                max_segment_num=20, max_frame_num=200,
                 slow_feature_dim = 128,
                 fast_feature_dim=256
                 ):
        nn.Module.__init__(self)
        self.device = device
        self.in_channel = in_channel
        self.conv1_channel = conv1_channel
        self.conv2_channel = conv2_channel
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.slow_conv_channel = slow_feature_dim
        self.fast_conv_channel = fast_feature_dim
        self.conv1d_slow = nn.Sequential (
                nn.Conv1d(in_channel, self.slow_conv_channel, kernel_size=5, stride=8, padding=2),
                nn.MaxPool1d(3,stride=1,padding=1),
                nn.BatchNorm1d(self.slow_conv_channel),
                nn.ReLU(),
                nn.Conv1d(self.slow_conv_channel, self.slow_conv_channel, kernel_size=5, stride=1, padding=2),
                nn.MaxPool1d(3, stride=2, padding=1)
                )
        self.conv1d_fast = nn.Sequential (
                nn.Conv1d(in_channel, self.fast_conv_channel, kernel_size=3, stride=2, padding=1),
                nn.MaxPool1d(2,stride=1,padding=0),
                nn.BatchNorm1d(self.fast_conv_channel),
                nn.ReLU(),
                nn.Conv1d(self.fast_conv_channel, self.fast_conv_channel, kernel_size=3, stride=1, padding=1),
                nn.MaxPool1d(3, stride=2, padding=1)
                )

        self.conv1d_1=nn.Conv1d(in_channel,self.conv1_channel, kernel_size=5,stride=1,padding=2)
        self.max_pooling_1=nn.MaxPool1d(2,stride=2,padding=0)
        self.conv1d_2=nn.Conv1d(self.conv1_channel,self.conv2_channel,kernel_size=5,stride=1,padding=2)
        self.max_pooling_2=nn.MaxPool1d(2,stride=2,padding=0)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_normal_(self.conv1d_1.weight)
        torch.nn.init.kaiming_normal_(self.conv1d_2.weight)

        torch.nn.init.kaiming_normal_(self.conv1d_slow[0].weight)
        torch.nn.init.kaiming_normal_(self.conv1d_slow[4].weight)

        torch.nn.init.kaiming_normal_(self.conv1d_fast[0].weight)
        torch.nn.init.kaiming_normal_(self.conv1d_fast[4].weight)

    # batch tensor: batch_size * max_seg_num * max_seg_length * 2048/4096
    # seg_len list(list(int)) : batch_size * seg_num (num of frame)
    # concept : batch_size * 300
    def forward(self, batch, frame_features,seg_len=None,concept1=None, concept2=None):
        batch_size=batch.size()[0]
        max_seg_num=batch.size()[1]
        max_seg_length=batch.size()[2]
        slow_features = self.conv1d_slow(frame_features.transpose(1,2))
        fast_features = self.conv1d_fast(frame_features.transpose(1, 2))
        # tmp1=self.conv1d_1(batch.view(batch_size*max_seg_num,max_seg_length,-1).transpose(1,2))
        # tmp1=self.max_pooling_1(tmp1)
        # tmp2=self.conv1d_2(tmp1)
        # tmp2=self.max_pooling_2(tmp2).transpose(1,2)
        return None, slow_features.transpose(1, 2), fast_features.transpose(1, 2)
