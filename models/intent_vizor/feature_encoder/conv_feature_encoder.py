import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(self, device="cuda",
                 in_channel=2048, conv1_channel=512, conv2_channel=256,
                 deconv1_channel=1024, deconv2_channel=1024, similarity_dim=1000,
                 concept_dim=300, max_segment_num=20, max_frame_num=200,
                 topic_num = 10, topic_embedding_dim=64, hidden_dim=64, output_mlp_hidden_dim=256):
        nn.Module.__init__(self)
        self.device = device
        self.in_channel = in_channel
        self.conv1_channel = conv1_channel
        self.conv2_channel = conv2_channel
        self.deconv1_channel = deconv1_channel
        self.deconv2_channel = deconv2_channel
        self.concept_dim = concept_dim
        self.similarity_dim = similarity_dim
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.convs_channel = conv1_channel
        self.conv1d_s = nn.Conv1d(in_channel, self.convs_channel, kernel_size=5, stride=8, padding=2)
        self.conv1d_1=nn.Conv1d(in_channel,self.conv1_channel, kernel_size=5,stride=1,padding=2)
        self.max_pooling_1=nn.MaxPool1d(2,stride=2,padding=0)
        self.conv1d_2=nn.Conv1d(self.conv1_channel,self.conv2_channel,kernel_size=5,stride=1,padding=2)
        self.max_pooling_2=nn.MaxPool1d(2,stride=2,padding=0)
        self.init_weight()

    def init_weight(self):
        pass
        # nn.init.normal_(self.topic_embedding.weight.data, 0, 0.001)

    # batch tensor: batch_size * max_seg_num * max_seg_length * 2048/4096
    # seg_len list(list(int)) : batch_size * seg_num (num of frame)
    # concept : batch_size * 300
    def forward(self,batch,seg_len,concept1, concept2):
        batch_size=batch.size()[0]
        max_seg_num=batch.size()[1]
        max_seg_length=batch.size()[2]
        tmp = self.conv1d_s(batch.view(batch_size*max_seg_num,max_seg_length,-1).transpose(1,2))
        print(tmp.size())
        # (batch_size * max_seg_num) * 128 * max_seg_length
        tmp1=self.conv1d_1(batch.view(batch_size*max_seg_num,max_seg_length,-1).transpose(1,2))
        # (batch_size * max_seg_num) * 128 * max_seg_length/2
        tmp1=self.max_pooling_1(tmp1)

        # (batch_size * max_seg_num) * 256 * max_seg_length/2
        tmp2=self.conv1d_2(tmp1)

        # (batch_size * max_seg_num) * max_seg_length/4 * 256
        tmp2=self.max_pooling_2(tmp2).transpose(1,2)

        return tmp2
