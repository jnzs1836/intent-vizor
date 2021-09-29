import torch.nn as nn
from models.intent_vizor.gcn.gtad import EgoGCNeXt


class SnippetTopicGCN(nn.Module):
    def __init__(self, temporal_scale=256, feature_dim=256, batch_size=4, topic_dim=16):
        super(SnippetTopicGCN, self).__init__()
        self.tscale = temporal_scale
        self.feat_dim = feature_dim
        self.bs = batch_size
        self.h_dim_1d = 256
        self.h_dim_2d = 128
        self.h_dim_3d = 512
        self.idx_list = []
        self.topic_dim = topic_dim

        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            # GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        )
        self.backbone_topic = nn.Sequential(
            nn.Conv1d(self.topic_dim, self.h_dim_1d, kernel_size=1, padding=0, groups=4),
            nn.ReLU(inplace=True)
        )
        self.gcn1 = EgoGCNeXt(self.h_dim_1d, self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list)

        self.gcn2 = EgoGCNeXt(self.h_dim_1d, self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list)

    def forward(self, snip_feature, seg_lens, topic_embedding):
        del self.idx_list[:]  # clean the idx list
        topic_embedding = topic_embedding.unsqueeze(-1)
        base_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        topic_feature = self.backbone_topic(topic_embedding).contiguous()
        base_feature = self.gcn1(base_feature, seg_lens, topic_feature)
        gcnext_feature = self.gcn2(base_feature, seg_lens, topic_feature)  #
        return gcnext_feature
