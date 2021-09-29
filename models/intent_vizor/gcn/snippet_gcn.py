import torch.nn as nn
from models.intent_vizor.gcn.gtad import GCNeXt


class SnippetGCN(nn.Module):
    def __init__(self, temporal_scale=256, feature_dim=256, batch_size=4):
        super(SnippetGCN, self).__init__()
        self.tscale = temporal_scale
        self.feat_dim = feature_dim
        self.bs = batch_size
        self.h_dim_1d = 256
        self.h_dim_2d = 128
        self.h_dim_3d = 512
        self.idx_list = []

        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            # GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        )
        self.gcn1 = GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list)

        # Regularization
        # self.regu_s = nn.Sequential(
        #     GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
        #     nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        # )
        # self.regu_e = nn.Sequential(
        #     GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
        #     nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        # )

        # Backbone Part 2
        # self.backbone2 = nn.Sequential(
        #     GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        # )
        self.gcn2 = GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list)




        # Position encoding (not used)
        # self.pos = torch.arange(0, 1, 1.0 / self.tscale).view(1, 1, self.tscale)

    def forward(self, snip_feature, seg_lens):
        del self.idx_list[:]  # clean the idx list
        base_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        base_feature = self.gcn1(base_feature, seg_lens)
        gcnext_feature = self.gcn2(base_feature, seg_lens)  #
        return gcnext_feature
